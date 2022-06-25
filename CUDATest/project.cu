#include "fluid_sim_cuda.cuh"

__global__ void ProjectKernel(HashMap<float>* velocity_x, HashMap<float>* velocity_y, HashMap<float>* velocity_z, HashMap<float>* data, HashMap<float>* data_prev, uint3 length) {
	unsigned int z_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;
	unsigned int x_bounds = blockIdx.z * blockDim.z + threadIdx.z + 1;

	if (threadIdx.x < length.x - 1 && threadIdx.y < length.y - 1 && threadIdx.z < length.z - 1) {
		IndexPair incident(z_bounds, y_bounds, x_bounds);

		data->Get(incident.IX(length.x)) =
			((velocity_x->Get(incident.Right().IX(length.x))
				- velocity_x->Get(incident.Left().IX(length.x))
				+ velocity_y->Get(incident.Up().IX(length.x))
				- velocity_y->Get(incident.Down().IX(length.x)))
				* -0.5f) / length.x;

		data_prev->Get(incident.IX(length.x)) = 0;
	}
	if (x_bounds == length.x - 1 && y_bounds == length.y - 1 && z_bounds == length.z - 1) {
		BoundaryConditions(0, data_prev, length);
		BoundaryConditions(0, data, length);
	}
}

__global__ void ProjectKernel2(HashMap<float>* velocity_x, HashMap<float>* velocity_y, HashMap<float>* velocity_z, HashMap<float>* data, HashMap<float>* data_prev, uint3 length) {
	unsigned int z_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;
	unsigned int x_bounds = blockIdx.z * blockDim.z + threadIdx.z + 1;

	if (threadIdx.x < length.x - 1 && threadIdx.y < length.y - 1 && threadIdx.z < length.z - 1) {
		IndexPair incident(z_bounds, y_bounds, x_bounds);
		float compute_x = velocity_x->Get(incident.IX(length.x)) - (-0.5f *
			(data_prev->Get(incident.Right().IX(length.x))
				- data_prev->Get(incident.Left().IX(length.x)))
			* length.x);

		float compute_y = velocity_y->Get(incident.IX(length.x)) - (-0.5f *
			(data_prev->Get(incident.Up().IX(length.x))
				- data_prev->Get(incident.Down().IX(length.x)))
			* length.x);

		float compute_z = velocity_z->Get(incident.IX(length.x)) - (-0.5f *
			(data_prev->Get(incident.Front().IX(length.x))
				- data_prev->Get(incident.Back().IX(length.x)))
			* length.x);

		velocity_x->Put(incident.IX(length.x), compute_x);
		velocity_y->Put(incident.IX(length.x), compute_y);
		velocity_z->Put(incident.IX(length.x), compute_z);
	}
	if (x_bounds == length.x - 1 && y_bounds == length.y - 1 && z_bounds == length.z - 1) {
		BoundaryConditions(1, velocity_x, length);
		BoundaryConditions(2, velocity_y, length);
	}
}

void ProjectCuda(int bounds, VectorField& velocity, VectorField& velocity_prev, const uint3& length, const unsigned int& iter) {
	cudaError_t cuda_status = cudaSuccess;

	dim3 blocks, threads;
	ThreadAllocator(blocks, threads, length.x);

	HashMap<float>* v_map_x = velocity.GetVectorMap()[0].map_->device_alloc_,
		*v_map_y = velocity.GetVectorMap()[1].map_->device_alloc_,
		*v_map_z = velocity.GetVectorMap()[2].map_->device_alloc_,
		*x_map = velocity_prev.GetVectorMap()[0].map_->device_alloc_,
		*y_map = velocity_prev.GetVectorMap()[1].map_->device_alloc_;

	ProjectKernel<<<blocks, threads>>> (v_map_x, v_map_y, v_map_z, x_map, y_map, length);
	LinearSolverKernel<<<blocks, threads>>> (x_map, y_map, 1, 4, length, iter, bounds);
	ProjectKernel2<<<blocks, threads>>> (v_map_x, v_map_y, v_map_z, x_map, y_map, length);

	std::cout << "Yo Pierre, you wanna come out here? *door squeaking noise*" << std::endl;

	cuda_status = PostExecutionChecks(cuda_status, "ProjectCudaKernel");
}