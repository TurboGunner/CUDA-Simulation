#include "fluid_sim_cuda.cuh"

__global__ void ProjectKernel(HashMap<float>* velocity_x, HashMap<float>* velocity_y, HashMap<float>* velocity_z, HashMap<float>* data, HashMap<float>* data_prev, uint3 length) {
	unsigned int z_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;
	unsigned int x_bounds = blockIdx.z * blockDim.z + threadIdx.z + 1;

	if (x_bounds < length.x - 1 && y_bounds < length.y - 1 && z_bounds < length.z - 1) {
		IndexPair incident(z_bounds, y_bounds, x_bounds);

		data_prev->Get(incident.IX(length.x)) = 0;

		data->Get(incident.IX(length.x)) =
			((velocity_x->Get(incident.Right().IX(length.x))
				- velocity_x->Get(incident.Left().IX(length.x))
				+ velocity_y->Get(incident.Up().IX(length.x))
				- velocity_y->Get(incident.Down().IX(length.x))
				+ velocity_z->Get(incident.Front().IX(length.x))
				- velocity_z->Get(incident.Back().IX(length.x)))
				* -0.5f) / length.x;
	}
}

__global__ void ProjectKernel2(HashMap<float>* velocity_x, HashMap<float>* velocity_y, HashMap<float>* velocity_z, HashMap<float>* data, HashMap<float>* data_prev, uint3 length) {
	unsigned int z_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;
	unsigned int x_bounds = blockIdx.z * blockDim.z + threadIdx.z + 1;

	if (x_bounds < length.x - 1 && y_bounds < length.y - 1 && z_bounds < length.z - 1) {
		IndexPair incident(z_bounds, y_bounds, x_bounds);
		float compute_x = velocity_x->Get(incident.IX(length.x)) - (0.5f *
			(data_prev->Get(incident.Right().IX(length.x))
				- data_prev->Get(incident.Left().IX(length.x)))
				* length.x);

		float compute_y = velocity_y->Get(incident.IX(length.x)) - (0.5f *
			(data_prev->Get(incident.Up().IX(length.x))
				- data_prev->Get(incident.Down().IX(length.x)))
				* length.x);

		float compute_z = velocity_z->Get(incident.IX(length.x)) - (0.5f *
			(data_prev->Get(incident.Front().IX(length.x))
				- data_prev->Get(incident.Back().IX(length.x)))
				* length.x);

		velocity_x->Put(incident.IX(length.x), compute_x);
		velocity_y->Put(incident.IX(length.x), compute_y);
		velocity_z->Put(incident.IX(length.x), compute_z);
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

	ProjectKernel<<<blocks, threads>>> (v_map_x, v_map_y, v_map_z, y_map, x_map, length);
	BoundaryConditionsCuda(0, x_map, length);
	BoundaryConditionsCuda(0, y_map, length);

	cuda_status = PostExecutionChecks(cuda_status, "ProjectFirstKernel");

	LinearSolverCuda(0, velocity_prev.GetVectorMap()[1], velocity_prev.GetVectorMap()[0], 1, 6, iter, length);

	cuda_status = PostExecutionChecks(cuda_status, "ProjectLinearSolve");

	ProjectKernel2<<<blocks, threads>>> (v_map_x, v_map_y, v_map_z, y_map, x_map, length);
	BoundaryConditionsCuda(1, v_map_x, length);
	BoundaryConditionsCuda(2, v_map_y, length);
	BoundaryConditionsCuda(3, v_map_z, length);

	std::cout << "Yo Pierre, you wanna come out here? *door squeaking noise*" << std::endl;

	cuda_status = PostExecutionChecks(cuda_status, "ProjectSecondKernel");
}