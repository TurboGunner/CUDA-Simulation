#include "fluid_sim_cuda.cuh"

__global__ void LinearSolverKernel(HashMap* data, HashMap* data_prev, float a_fac, float c_fac, uint3 length, unsigned int iter, int bounds) {
	unsigned int z_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;
	unsigned int x_bounds = blockIdx.z * blockDim.z + threadIdx.z + 1;

	IndexPair incident(x_bounds, y_bounds, z_bounds);
	float temp = data->Get(incident.IX(length.x));
	for (int i = 0; i < iter; i++) {
		float compute = data_prev->Get(incident.IX(length.x)) +
			a_fac *
			(data->Get(incident.Right().IX(length.x))
				+ data->Get(incident.Left().IX(length.x))
				+ data->Get(incident.Up().IX(length.x))
				+ data->Get(incident.Down().IX(length.x))
				+ data->Get(incident.Front().IX(length.x))
				+ data->Get(incident.Back().IX(length.x)))
			* (1.0f / c_fac);

		data->Get(incident.IX(length.x)) = compute;
	}
}

cudaError_t LinearSolverCuda(int bounds, AxisData& current, AxisData& previous, const float& a_fac, const float& c_fac, const unsigned int& iter, const uint3& length) {
	cudaError_t cuda_status = cudaSuccess;

	dim3 blocks, threads;
	ThreadAllocator(blocks, threads, length.x);

	HashMap* c_map = current.map_->device_alloc_,
		*p_map = previous.map_->device_alloc_;

	LinearSolverKernel<<<blocks, threads>>> (current.map_->device_alloc_, p_map, a_fac, c_fac, length, iter, bounds);

	BoundaryConditionsCuda(bounds, current, length);

	cuda_status = PostExecutionChecks(cuda_status, "LinearSolverKernel");
	return cuda_status;
}