#include "fluid_sim_cuda.cuh"

__global__ void AdvectKernel(HashMap<float>* data, HashMap<float>* data_prev, HashMap<float>* velocity_x, HashMap<float>* velocity_y, float dt, unsigned int length, int bounds) {
	unsigned int y_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int x_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;

	float x_current, x_previous, y_current, y_previous;

	float x_dt = dt * (length - 2), y_dt = dt * (length - 2);

	float density_current, density_prev, time_current, time_prev;
	float x_value, y_value;

	if (threadIdx.x < length - 1 && threadIdx.y < length - 1) {
		x_value = (float)x_bounds - (x_dt * (*velocity_x)[IndexPair(y_bounds, x_bounds).IX(length)]);
		y_value = (float)y_bounds - (y_dt * (*velocity_y)[IndexPair(y_bounds, x_bounds).IX(length)]);

		if (x_value < 0.5f) {
			x_value = 0.5f;
		}
		if (x_value > length + 0.5f) {
			x_value = length + 0.5f;
		}
		x_current = floorf(x_value);
		x_previous = x_current + 1.0f;
		if (y_value < 0.5f) {
			y_value = 0.5f;
		}
		if (y_value > length + 0.5f) {
			y_value = length + 0.5f;
		}
		y_current = floorf(y_value);
		y_previous = y_current + 1.0f;

		density_prev = x_value - x_current;
		density_current = 1.0f - density_prev;
		time_prev = y_value - y_current;
		time_current = 1.0f - time_prev;

		(*data)[IndexPair(y_bounds, x_bounds).IX(length)] =
			(density_current * 
				(((*data_prev)[IndexPair(x_current, y_current).IX(length)] * time_current) +
				((*data_prev)[IndexPair(x_current, y_previous).IX(length)] * time_prev))) +
			(density_prev *
				(((*data_prev)[IndexPair(x_previous, y_current).IX(length)] * time_current) +
				((*data_prev)[IndexPair(x_previous, y_previous).IX(length)] * time_prev)));
	}
	if (x_bounds == length - 1 && y_bounds == length - 1) {
		BoundaryConditions(bounds, data, length);
	}
}

void AdvectCuda(int bounds, AxisData& current, AxisData& previous, VectorField& velocity, const float& dt, const unsigned int& length) {
	unsigned int alloc_size = length * length;

	cudaError_t cuda_status = cudaSuccess;

	dim3 blocks, threads;
	ThreadAllocator(blocks, threads, length);

	HashMap<float>* v_map_x = velocity.GetVectorMap()[0].map_->device_alloc_,
		*v_map_y = velocity.GetVectorMap()[1].map_->device_alloc_,
		*c_map = current.map_->device_alloc_,
		*p_map = previous.map_->device_alloc_;

	std::cout << "bidoof" << std::endl;

	AdvectKernel<<<blocks, threads>>> (c_map, p_map, v_map_x, v_map_y, dt, length, bounds);

	PostExecutionChecks(cuda_status, "AdvectCudaKernel");
}