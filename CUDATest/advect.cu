#include "fluid_sim_cuda.cuh"

__global__ void AdvectKernel(HashMap<IndexPair, float, HashDupe<IndexPair>>* data, HashMap<IndexPair, float, HashDupe<IndexPair>>* data_prev, HashMap<IndexPair, F_Vector, Hash<IndexPair>>* velocity, float dt, unsigned int length, int bounds) {
	unsigned int y_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int x_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;

	float x_current, x_previous, y_current, y_previous;

	float x_dt = dt * (length - 2), y_dt = dt * (length - 2);

	float velocity_x_curr, velocity_x_prev, velocity_y_curr, velocity_y_prev;
	float x_value, y_value;

	if (threadIdx.x < length - 1 && threadIdx.y < length - 1) {
		x_value = (float)x_bounds - (x_dt * (*velocity)[IndexPair(y_bounds, x_bounds)].vx_);
		y_value = (float)y_bounds - (y_dt * (*velocity)[IndexPair(y_bounds, x_bounds)].vy_);

		if (x_value < 0.5f) {
			x_value = 0.5f;
		}
		if (x_value > length + 0.5f) {
			x_value = length + 0.5f;
		}
		x_current = x_value;
		x_previous = x_current + 1.0f;
		if (y_value < 0.5f) {
			y_value = 0.5f;
		}
		if (y_value > length + 0.5f) {
			y_value = length + 0.5f;
		}
		y_current = y_value;
		y_previous = y_current + 1.0f;

		velocity_x_prev = x_value - x_current;
		velocity_x_curr = 1.0f - velocity_x_prev;
		velocity_y_prev = y_value - y_current;
		velocity_y_curr = 1.0f - velocity_y_prev;

		(*data)[IndexPair(x_current, y_current)] =
			(((*data_prev)[IndexPair(x_current, y_current)] * velocity_y_curr) +
				((*data_prev)[IndexPair(x_current, y_previous)] * velocity_y_prev) * velocity_x_curr) +
			(((*data_prev)[IndexPair(x_previous, y_current)] * velocity_y_curr) +
				((*data_prev)[IndexPair(x_previous, y_previous)] * velocity_y_prev) * velocity_x_prev);
	}
	if (x_bounds * y_bounds >= (length * length)) {
		BoundaryConditions(bounds, data, length);
	}
}

void AdvectCuda(int bounds, AxisData& current, AxisData& previous, VectorField& velocity, const float& dt, const unsigned int& length) {
	unsigned int alloc_size = length * length;
	CudaMethodHandler handler(alloc_size, "AdvectCudaKernel");

	cudaError_t cuda_status = cudaSuccess;

	dim3 blocks, threads;
	ThreadAllocator(blocks, threads, length);

	AdvectKernel<<<blocks, threads>>> (current.map_, previous.map_, velocity.GetVectorMap(), dt, length, bounds);

	cuda_status = handler.PostExecutionChecks(cuda_status);
}