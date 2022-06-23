#include "fluid_sim_cuda.cuh"

__global__ void ProjectKernel(HashMap<float>* velocity_x, HashMap<float>* velocity_y, HashMap<float>* data, HashMap<float>* data_prev, unsigned int length) {
	unsigned int y_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int x_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;

	if (threadIdx.x < length - 1 && threadIdx.y < length - 1) { 
		IndexPair incident(y_bounds, x_bounds);
		data->Get(incident.IX(length)) =
			((velocity_x->Get(incident.Right().IX(length))
				- velocity_x->Get(incident.Left().IX(length))
				+ velocity_y->Get(incident.Up().IX(length))
				- velocity_y->Get(incident.Down().IX(length)))
				* -0.5f) / length;
		data_prev->Get(incident.IX(length)) = 0;
	}
	if (x_bounds * y_bounds >= (length * length)) {
		BoundaryConditions(0, data, length);
		BoundaryConditions(0, data_prev, length);
	}
}

__global__ void ProjectKernel2(HashMap<float>* velocity_x, HashMap<float>* velocity_y, HashMap<float>* data, HashMap<float>* data_prev, unsigned int length) {
	unsigned int y_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int x_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;

	if (threadIdx.x < length - 1 && threadIdx.y < length - 1) {
		IndexPair incident(y_bounds, x_bounds);
		float compute_x = velocity_x->Get(incident.IX(length)) - (-0.5f *
			(data_prev->Get(incident.Right().IX(length))
				- data_prev->Get(incident.Left().IX(length)))
			* length);

		float compute_y = velocity_y->Get(incident.IX(length)) - (-0.5f *
			(data_prev->Get(incident.Up().IX(length))
				- data_prev->Get(incident.Down().IX(length)))
			* length);
		velocity_x->Put(incident.IX(length), compute_x);
		velocity_y->Put(incident.IX(length), compute_y);
	}
	if (x_bounds * y_bounds >= (length * length)) {
		BoundaryConditions(1, velocity_x, length);
		BoundaryConditions(2, velocity_y, length);
	}
}

void ProjectCuda(int bounds, VectorField& velocity, VectorField& velocity_prev, const unsigned int& length, const unsigned int& iter) {
	unsigned int alloc_size = length * length;

	cudaError_t cuda_status = cudaSuccess;

	dim3 blocks, threads;
	ThreadAllocator(blocks, threads, length);

	HashMap<float>* v_map_x = velocity.GetVectorMap()[0].map_->device_alloc_,
		*v_map_y = velocity.GetVectorMap()[1].map_->device_alloc_,
		*x_map = velocity_prev.GetVectorMap()[0].map_->device_alloc_,
		*y_map = velocity_prev.GetVectorMap()[1].map_->device_alloc_;

	ProjectKernel<<<blocks, threads>>> (v_map_x, v_map_y, x_map, y_map, length);
	LinearSolverKernel<<<blocks, threads>>> (x_map, y_map, 1, 4, length, iter, bounds);
	ProjectKernel2<<<blocks, threads>>> (v_map_x, v_map_y, x_map, y_map, length);

	std::cout << "Yo Pierre, you wanna come out here? *door squeaking noise*" << std::endl;

	PostExecutionChecks(cuda_status, "ProjectCudaKernel");
}