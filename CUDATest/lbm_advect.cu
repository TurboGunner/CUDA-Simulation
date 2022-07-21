#include "fluid_sim_cuda.cuh"
#include "lbm_sim_cuda.cuh"

__device__ inline float EquilibriumFunction(HashMap* velocity_x, HashMap* velocity_y, HashMap* velocity_z, float total_density, IndexPair incident, float3 total_v, uint3 length) {

	float shared_step = (velocity_x->Get(incident.IX(length.x)) * total_v.x +
		velocity_y->Get(incident.IX(length.x)) * total_v.y +
		velocity_z->Get(incident.IX(length.x)) * total_v.z);

	float intermediate_1 = 1.0f + (3.0f * shared_step);

	float intermediate_2 = (9.0f / 2.0f) * (shared_step * shared_step);

	float intermediate_3 = (3.0f / 2.0f) * powf(velocity_x->Get(incident.IX(length.x)), 2)
		+ powf(velocity_y->Get(incident.IX(length.x)), 2)
		+ powf(velocity_z->Get(incident.IX(length.x)), 2);

	return total_density * incident.weight * (intermediate_1 + intermediate_2 + intermediate_3);
}

__global__ void LBMAdvectKernel(HashMap* data, HashMap* velocity_x, HashMap* velocity_y, HashMap* velocity_z, float total_density, float visc, float dt, float3 total_v, uint3 length) {
	unsigned int z_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;
	unsigned int x_bounds = blockIdx.z * blockDim.z + threadIdx.z + 1;

	IndexPair incident(x_bounds, y_bounds, z_bounds);

	IndexPair incidents[19] = { incident,
		incident.Left(), incident.Right(), incident.Front(), incident.Back(), incident.Up(), incident.Down(),
		incident.CornerLDownFront(), incident.CornerLDownBack(), incident.CornerRDownFront(), incident.CornerRDownBack(),
		incident.CornerLUpFront(), incident.CornerLUpBack(), incident.CornerRUpFront(), incident.CornerRUpBack(),
		incident.CornerLMidBack(), incident.CornerRMidBack(), incident.CornerLMidFront(), incident.CornerRMidFront()
	};

	for (size_t i = 0; i < 19; i++) {
		float f_eq = EquilibriumFunction(velocity_x, velocity_y, velocity_z, total_density, incidents[i], total_v, length);

		data->Get(incidents[i].IX(length.x)) += -(1.0f / visc) * (data->Get(incidents[i].IX(length.x)) - f_eq);
	}
}

cudaError_t LBMAdvectCuda(int bounds, AxisData& current, VectorField& velocity, const float& visc, const float& dt, const uint3& length) {
	cudaError_t cuda_status = cudaSuccess;

	dim3 blocks, threads;
	ThreadAllocator(blocks, threads, length.x);

	HashMap* v_map_x = velocity.map_[0].map_->device_alloc_,
		*v_map_y = velocity.map_[1].map_->device_alloc_,
		*v_map_z = velocity.map_[2].map_->device_alloc_,
		*c_map = current.map_->device_alloc_,

	LBMAdvectKernel<<<blocks, threads>>> (c_map, v_map_x, v_map_y, v_map_z, visc, current.total_, dt, length);

	BoundaryConditionsCuda(bounds, current, length);

	cuda_status = PostExecutionChecks(cuda_status, "LBMStreamKernel");
}