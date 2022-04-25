#include "fluid_sim_cuda.cuh"

__global__ void AdvectKernel(float* result_ptr, float* data, float* data_prev, float3* velocity, float dt, unsigned int length) {
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;

	float x_current, x_previous, y_current, y_previous;

	float x_dt = dt * (length - 2);
	float y_dt = dt * (length - 2);

	float velocity_x_curr, velocity_x_prev, velocity_y_curr, velocity_y_prev;

	float x_value, y_value;

	if (threadIdx.x < length - 1 && threadIdx.y < length - 1) {
		x_value = (float) x_bounds - (x_dt * velocity[IX(x_bounds, y_bounds + 1, length)].x);
		y_value = (float) y_bounds - (y_dt * velocity[IX(x_bounds, y_bounds + 1, length)].y);

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
	}
	result_ptr[IX(x_bounds, y_bounds + 1, length)] = data[IX(x_bounds, y_bounds + 1, length)];
	if (x_bounds * y_bounds >= (length * length)) {
		PointerBoundaries(result_ptr, length);
	}
}

float* AdvectCuda(int bounds, VectorField& current, VectorField& previous, VectorField& velocity, const float& dt, const unsigned int& length) {
	float* curr_copy_ptr = nullptr, *prev_copy_ptr = nullptr;

	float* current_ptr = current.FlattenMapX(), //Maybe make current and previous part of the same vector to consolidate?
		*prev_ptr = previous.FlattenMapX();

	float3* v_ptr = velocity.FlattenMap(),
		*v_copy_ptr = nullptr;

	unsigned int alloc_size = length * length;

	float* result_ptr = new float[alloc_size],
		*result_copy_ptr = nullptr;

	vector<reference_wrapper<float*>> bidoof;
	bidoof.insert(bidoof.end(), { curr_copy_ptr, prev_copy_ptr, result_copy_ptr });

	CudaMemoryAllocator(bidoof, (size_t) alloc_size, sizeof(float));

	vector<reference_wrapper<float3*>> bidoof2;
	bidoof2.insert(bidoof2.end(), { v_copy_ptr } );

	//Maybe have more consolidation by auto-copying allocated pointers to the GPU
	CudaMemoryAllocator(bidoof2, (size_t) alloc_size, sizeof(float3));

	cudaError_t cuda_status = cudaSuccess;

	cuda_status = CopyFunction("cudaMemcpy failed!", curr_copy_ptr, current_ptr,
		cudaMemcpyHostToDevice, cuda_status, (size_t) alloc_size,
		sizeof(float));

	cuda_status = CopyFunction("cudaMemcpy failed!", prev_copy_ptr, prev_ptr,
		cudaMemcpyHostToDevice, cuda_status, (size_t) alloc_size,
		sizeof(float));

	cuda_status = CopyFunction("cudaMemcpy failed!", v_copy_ptr, v_ptr,
		cudaMemcpyHostToDevice, cuda_status, (size_t) alloc_size,
		sizeof(float3));

	dim3 blocks, threads;
	ThreadAllocator(blocks, threads, length);

	AdvectKernel<<<blocks, threads>>> (result_copy_ptr, curr_copy_ptr, prev_copy_ptr, v_copy_ptr, dt, length);

	function<cudaError_t()> error_check_func = []() { return cudaGetLastError(); };
	cuda_status = WrapperFunction(error_check_func, "cudaGetLastError (kernel launch)", "LinearSolverKernel", cuda_status);

	function<cudaError_t()> sync_func = []() { return cudaDeviceSynchronize(); };
	cuda_status = WrapperFunction(sync_func, "cudaDeviceSynchronize", "LinearSolverKernel", cuda_status);

	cuda_status = CopyFunction("cudaMemcpy failed!", result_ptr, result_copy_ptr,
		cudaMemcpyDeviceToHost, cuda_status, (size_t)alloc_size,
		sizeof(float));

	if (cuda_status != cudaSuccess) {
		CudaMemoryFreer(bidoof);
	}
	return result_ptr;
}