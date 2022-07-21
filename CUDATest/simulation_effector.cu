#include "fluid_sim_cuda.cuh"
#include "lbm_sim_cuda.cuh"

__global__ void AddOnAxis(HashMap* data, IndexPair origin, float value, uint3 length) {
	data->Get(origin.IX(length.x)) += value;
}

cudaError_t AddOnAxisCuda(AxisData& current, const IndexPair& origin, const float& value, const uint3& length) {
	cudaError_t cuda_status = cudaSuccess;

	HashMap*& c_map = current.map_->device_alloc_;

	AddOnAxis<<<1, 1>>> (c_map, origin, value, length);

	cuda_status = PostExecutionChecks(cuda_status, "AddOnAxisKernel");
	return cuda_status;
}

__global__ void AddOnVector(HashMap* velocity_x, HashMap* velocity_y, HashMap* velocity_z, IndexPair origin, float3 value, uint3 length) {
	velocity_x->Get(origin.IX(length.x)) += value.x;
	velocity_y->Get(origin.IX(length.x)) += value.y;
	velocity_z->Get(origin.IX(length.x)) += value.z;
}

cudaError_t AddOnVectorCuda(VectorField& velocity, const IndexPair& origin, const float3& value, const uint3& length) {
	cudaError_t cuda_status = cudaSuccess;

	HashMap*& v_map_x = velocity.map_[0].map_->device_alloc_,
		*&v_map_y = velocity.map_[1].map_->device_alloc_,
		*&v_map_z = velocity.map_[2].map_->device_alloc_;

	AddOnVector<<<1, 1>>> (v_map_x, v_map_y, v_map_z, origin, value, length);

	cuda_status = PostExecutionChecks(cuda_status, "AddOnVectorKernel");
	return cuda_status;
}

__global__ void VectorNormalKernel(HashMap* velocity_x, HashMap* velocity_y, HashMap* velocity_z, uint3 length) {
	unsigned int z_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;
	unsigned int x_bounds = blockIdx.z * blockDim.z + threadIdx.z + 1;

	IndexPair incident(x_bounds, y_bounds, z_bounds);

	unsigned int v_length = velocity_x->Get(incident.IX(length.x)) +
		velocity_y->Get(incident.IX(length.x)) +
		velocity_z->Get(incident.IX(length.x));

	velocity_x->Get(incident.IX(length.x)) = powf(velocity_x->Get(incident.IX(length.x)), 2) / v_length;
	velocity_y->Get(incident.IX(length.x)) = powf(velocity_y->Get(incident.IX(length.x)), 2) / v_length;
	velocity_z->Get(incident.IX(length.x)) = powf(velocity_z->Get(incident.IX(length.x)), 2) / v_length;
}

cudaError_t VectorNormalCuda(VectorField& velocity, const uint3& length) {
	cudaError_t cuda_status = cudaSuccess;

	dim3 blocks, threads;
	ThreadAllocator(blocks, threads, length.x);

	HashMap*& v_map_x = velocity.map_[0].map_->device_alloc_,
		*& v_map_y = velocity.map_[1].map_->device_alloc_,
		*& v_map_z = velocity.map_[2].map_->device_alloc_;

	VectorNormalKernel<<<blocks, threads>>> (v_map_x, v_map_y, v_map_z, length);

	cuda_status = PostExecutionChecks(cuda_status, "AddOnVectorKernel");
	return cuda_status;
}