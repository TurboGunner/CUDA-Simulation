#include "fluid_sim_cuda.cuh"

__global__ void ProjectKernel(HashMap* velocity_x, HashMap* velocity_y, HashMap* velocity_z, HashMap* data, HashMap* data_prev, uint3 length) {
	unsigned int z_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;
	unsigned int x_bounds = blockIdx.z * blockDim.z + threadIdx.z + 1;

	IndexPair incident(x_bounds, y_bounds, z_bounds);

	data->Get(incident.IX(length.x)) = 0;

	data_prev->Get(incident.IX(length.x)) = -0.5f *
		(velocity_x->Get(incident.Right().IX(length.x))
		- velocity_x->Get(incident.Left().IX(length.x))
		+ velocity_y->Get(incident.Up().IX(length.x))
		- velocity_y->Get(incident.Down().IX(length.x))
		+ velocity_z->Get(incident.Front().IX(length.x))
		- velocity_z->Get(incident.Back().IX(length.x)))
		/ length.x;
}

__global__ void ProjectKernel2(HashMap* velocity_x, HashMap* velocity_y, HashMap* velocity_z, HashMap* data, HashMap* data_prev, uint3 length) {
	unsigned int z_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;
	unsigned int x_bounds = blockIdx.z * blockDim.z + threadIdx.z + 1;

	IndexPair incident(x_bounds, y_bounds, z_bounds);

	velocity_x->Get(incident.IX(length.x)) -= (0.5f *
		(data->Get(incident.Right().IX(length.x) )
		- data->Get(incident.Left().IX(length.x)))
		* length.x);

	velocity_y->Get(incident.IX(length.x)) -= (0.5f *
		(data->Get(incident.Up().IX(length.x))
		- data->Get(incident.Down().IX(length.x)))
		* length.x);

	velocity_z->Get(incident.IX(length.x)) -= (0.5f *
		(data->Get(incident.Front().IX(length.x))
		- data->Get(incident.Back().IX(length.x)))
		* length.x);
}

cudaError_t ProjectCuda(int bounds, VectorField& velocity, VectorField& velocity_prev, const uint3& length, const unsigned int& iter) {
	cudaError_t cuda_status = cudaSuccess;

	dim3 blocks, threads;
	ThreadAllocator(blocks, threads, length.x);

	HashMap*& v_map_x = velocity.map_[0].map_->device_alloc_,
		*&v_map_y = velocity.map_[1].map_->device_alloc_,
		*&v_map_z = velocity.map_[2].map_->device_alloc_,
		*&x_map = velocity_prev.map_[0].map_->device_alloc_,
		*&y_map = velocity_prev.map_[1].map_->device_alloc_;

	ProjectKernel<<<blocks, threads>>> (v_map_x, v_map_y, v_map_z, x_map, y_map, length);
	BoundaryConditionsCuda(0, x_map, length);
	BoundaryConditionsCuda(0, y_map, length);

	cuda_status = PostExecutionChecks(cuda_status, "ProjectFirstKernel");

	LinearSolverCuda(0, velocity_prev.map_[0], velocity_prev.map_[1], 1, 6, iter, length);

	cuda_status = PostExecutionChecks(cuda_status, "ProjectLinearSolve");

	ProjectKernel2<<<blocks, threads>>> (v_map_x, v_map_y, v_map_z, x_map, y_map, length);
	BoundaryConditionsCuda(1, v_map_x, length);
	BoundaryConditionsCuda(2, v_map_y, length);
	BoundaryConditionsCuda(3, v_map_z, length);

	std::cout << "Yo Pierre, you wanna come out here? *door squeaking noise*" << std::endl;

	cuda_status = PostExecutionChecks(cuda_status, "ProjectSecondKernel");

	return cuda_status;
}