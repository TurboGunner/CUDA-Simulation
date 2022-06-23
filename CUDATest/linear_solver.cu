#include "fluid_sim_cuda.cuh"

__global__ void LinearSolverKernel(HashMap<float>* data, HashMap<float>* data_prev, float a_fac, float c_fac, unsigned int length, unsigned int iter, int bounds) {
	unsigned int y_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int x_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;

	if (threadIdx.x < length - 1 && threadIdx.y < length - 1) {
		IndexPair incident(y_bounds, x_bounds);
		for (int i = 0; i < iter; i++) {
			float compute = data_prev->Get(incident.IX(length)) +
				a_fac *
				(data->Get(incident.Right().IX(length))
				+ data->Get(incident.Left().IX(length))
				+ data->Get(incident.Up().IX(length))
				+ data->Get(incident.Down().IX(length)))
				* (1.0f / c_fac);
			data->Get(incident.IX(length)) = compute;
		}
	}
	if (x_bounds * y_bounds >= (length * length)) {
		BoundaryConditions(bounds, data, length);
	}
}

void LinearSolverCuda(int bounds, AxisData& current, AxisData& previous, const float& a_fac, const float& c_fac, const unsigned int& iter, const unsigned int& length) {
	cudaError_t cuda_status = cudaSuccess;

	dim3 blocks, threads;
	ThreadAllocator(blocks, threads, length);

	HashMap<float>* c_map = current.map_->device_alloc_,
		*p_map = previous.map_->device_alloc_;

	LinearSolverKernel<<<blocks, threads>>> (c_map, p_map, a_fac, c_fac, length, iter, bounds);

	PostExecutionChecks(cuda_status, "ProjectCudaKernel");

	cuda_status = PostExecutionChecks(cuda_status, "LinearSolverKernel");
}