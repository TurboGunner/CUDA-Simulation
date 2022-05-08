#include "fluid_sim_cuda.cuh"

__global__ void AdvectKernel(float* data, float* data_prev, float3* velocity, float dt, unsigned int length) {
	unsigned int y_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int x_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;

	float x_current, x_previous, y_current, y_previous;

	float x_dt = dt * (length - 2), y_dt = dt * (length - 2);

	float velocity_x_curr, velocity_x_prev, velocity_y_curr, velocity_y_prev;
	float x_value, y_value;

	if (threadIdx.x < length - 1 && threadIdx.y < length - 1) {
		x_value = (float)x_bounds - (x_dt * velocity[IX(x_bounds, y_bounds + 1, length)].x);
		y_value = (float)y_bounds - (y_dt * velocity[IX(x_bounds, y_bounds + 1, length)].y);

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

		data[IX(x_bounds, y_bounds + 1, length)] =
			((data_prev[IX(unsigned int(x_current), unsigned int(y_current + 1), length)] * velocity_y_curr) +
				(data_prev[IX(unsigned int(x_current), int(y_previous + 1), length)] * velocity_y_prev) * velocity_x_curr) +
			((data_prev[IX(unsigned int(x_previous), unsigned int(y_current + 1), length)] * velocity_y_curr) +
				(data_prev[IX(unsigned int(x_previous), unsigned int(y_previous + 1), length)] * velocity_y_prev) * velocity_x_prev);

		printf("%.5f | %d ", (data_prev[IX(unsigned int(x_current), unsigned int(y_current + 1), length)]), x_bounds);
	}
	if (x_bounds * y_bounds >= (length * length)) {
		PointerBoundaries(data, length);
	}
}

void AdvectCuda(int bounds, VectorField& current, VectorField& previous, VectorField& velocity, const float& dt, const unsigned int& length) {
	unsigned int alloc_size = length * length;
	CudaMethodHandler handler(alloc_size, "AdvectCudaKernel");
	float* curr_copy_ptr = nullptr, * prev_copy_ptr = nullptr;

	float* current_ptr = current.FlattenMapX(), //Maybe make current and previous part of the same vector to consolidate?
		* prev_ptr = previous.FlattenMapX();

	float3* v_ptr = velocity.FlattenMap(), * v_copy_ptr = nullptr;

	handler.float_copy_ptrs_.insert(handler.float_copy_ptrs_.end(), { curr_copy_ptr, prev_copy_ptr });
	handler.float_ptrs_.insert(handler.float_ptrs_.end(), { current_ptr, prev_ptr });

	handler.float3_copy_ptrs_.insert(handler.float3_copy_ptrs_.end(), { v_copy_ptr });
	handler.float3_ptrs_.insert(handler.float3_ptrs_.end(), { v_ptr });

	handler.AllocateCopyPointers();

	cudaError_t cuda_status = cudaSuccess;

	cuda_status = handler.CopyToMemory(cudaMemcpyHostToDevice, cuda_status);

	dim3 blocks, threads;
	ThreadAllocator(blocks, threads, length);

	AdvectKernel <<<blocks, threads>>> (curr_copy_ptr, prev_copy_ptr, v_copy_ptr, dt, length);

	cuda_status = handler.PostExecutionChecks(cuda_status);

	cuda_status = CopyFunction("cudaMemcpy failed!", current_ptr, curr_copy_ptr,
		cudaMemcpyDeviceToHost, cuda_status, (size_t)alloc_size,
		sizeof(float));

	current.RepackMap(current_ptr, curr_copy_ptr);
}