#include "fluid_sim_cuda.cuh"

__global__ void AdvectKernel(HashMap<float>* data, HashMap<float>* data_prev, HashMap<F_Vector>* velocity, float dt, unsigned int length, int bounds) {
	unsigned int y_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int x_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;

	float x_current, x_previous, y_current, y_previous;

	float x_dt = dt * (length - 2), y_dt = dt * (length - 2);

	float velocity_x_curr, velocity_x_prev, velocity_y_curr, velocity_y_prev;
	float x_value, y_value;

	if (threadIdx.x < length - 1 && threadIdx.y < length - 1) {
		x_value = (float)x_bounds - (x_dt * (*velocity)[IndexPair(y_bounds, x_bounds).IX(length)].vx_);
		y_value = (float)y_bounds - (y_dt * (*velocity)[IndexPair(y_bounds, x_bounds).IX(length)].vy_);

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

		(*data)[IndexPair(x_current, y_current).IX(length)] =
			(((*data_prev)[IndexPair(x_current, y_current).IX(length)] * velocity_y_curr) +
				((*data_prev)[IndexPair(x_current, y_previous).IX(length)] * velocity_y_prev) * velocity_x_curr) +
			(((*data_prev)[IndexPair(x_previous, y_current).IX(length)] * velocity_y_curr) +
				((*data_prev)[IndexPair(x_previous, y_previous).IX(length)] * velocity_y_prev) * velocity_x_prev);
	}
	if (x_bounds * y_bounds >= (length * length)) {
		BoundaryConditions(bounds, data, length);
	}
}

void AdvectCuda(int bounds, AxisData& current, AxisData& previous, VectorField& velocity, const float& dt, const unsigned int& length) {
	unsigned int alloc_size = length * length;

	cudaError_t cuda_status = cudaSuccess;

	dim3 blocks, threads;
	ThreadAllocator(blocks, threads, length);

	HashMap<F_Vector>* v_map = nullptr;
	HashMap<float>* c_map = nullptr, *p_map = nullptr;

	velocity.GetVectorMap()->DeviceTransfer(cuda_status, velocity.GetVectorMap(), v_map);
	current.map_->DeviceTransfer(cuda_status, current.map_, c_map);
	previous.map_->DeviceTransfer(cuda_status, previous.map_, p_map);

	std::cout << "bidoof" << std::endl;

	AdvectKernel<<<blocks, threads>>> (c_map, p_map, v_map, dt, length, bounds);

	PostExecutionChecks(cuda_status, "ProjectCudaKernel");

	velocity.GetVectorMap()->HostTransfer(cuda_status);
	current.map_->HostTransfer(cuda_status);
	previous.map_->HostTransfer(cuda_status);
}