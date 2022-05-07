#include "fluid_sim_cuda.cuh"

__global__ void ProjectKernel(float3* result_ptr, float* data, float* data_prev, float3* velocity, unsigned int length, unsigned int iter, int bounds) {
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;

	if (threadIdx.x < length - 1 && threadIdx.y < length - 1) {
		data[IX(x_bounds, y_bounds + 1, length)] =
			(velocity[IX(x_bounds + 1, y_bounds + 1, length)].x
				- velocity[IX(x_bounds - 1, y_bounds + 1, length)].x
				+ velocity[IX(x_bounds, y_bounds + 2, length)].y
				- velocity[IX(x_bounds, y_bounds, length)].y)
			* -0.5f * (1.0f / length);

		data_prev[IX(x_bounds, y_bounds + 1, length)] = 0;
	}
	printf("%.5f\n", data[IX(x_bounds, y_bounds + 1, length)]);
	if (x_bounds * y_bounds >= (length * length)) {
		PointerBoundaries(data, length);
		PointerBoundaries(data_prev, length);
		LinearSolverGPU(data, data_prev, 1, 4, length, iter, bounds);
	}

	if (threadIdx.x < length - 1 && threadIdx.y < length - 1) {
		result_ptr[IX(x_bounds, y_bounds + 1, length)].x -= 0.5f
			* (data_prev[IX(x_bounds + 1, y_bounds + 1, length)]
			- data_prev[IX(x_bounds - 1, y_bounds + 1, length)]) 
			* length;
		result_ptr[IX(x_bounds, y_bounds + 1, length)].y -= 0.5f
			* (data_prev[IX(x_bounds, y_bounds + 2, length)]
			- data_prev[IX(x_bounds, y_bounds, length)])
			* length;
	}
	if (x_bounds * y_bounds >= (length * length)) {
		PointerBoundariesSpecial(velocity, length);
	}
}

tuple<float3*, float*, float*> ProjectCuda(int bounds, VectorField& current, VectorField& previous, VectorField& velocity, const unsigned int& length, const unsigned int& iter) {
	float* curr_copy_ptr = nullptr, * prev_copy_ptr = nullptr;

	float* current_ptr = current.FlattenMapX(), //Maybe make current and previous part of the same vector to consolidate?
		* prev_ptr = previous.FlattenMapX();

	float3* v_ptr = velocity.FlattenMap(),
		* v_copy_ptr = nullptr;

	unsigned int alloc_size = length * length;

	float3* result_ptr = new float3[alloc_size],
		*result_copy_ptr = nullptr;

	vector<reference_wrapper<float*>> bidoof;
	bidoof.insert(bidoof.end(), { curr_copy_ptr, prev_copy_ptr });

	CudaMemoryAllocator(bidoof, (size_t)alloc_size, sizeof(float));

	vector<reference_wrapper<float3*>> bidoof2;
	bidoof2.insert(bidoof2.end(), { v_copy_ptr, result_copy_ptr });

	//Maybe have more consolidation by auto-copying allocated pointers to the GPU
	CudaMemoryAllocator(bidoof2, (size_t)alloc_size, sizeof(float3));

	cudaError_t cuda_status = cudaSuccess;

	cuda_status = CopyFunction("cudaMemcpy failed!", curr_copy_ptr, current_ptr,
		cudaMemcpyHostToDevice, cuda_status, (size_t)alloc_size,
		sizeof(float));

	cuda_status = CopyFunction("cudaMemcpy failed!", prev_copy_ptr, prev_ptr,
		cudaMemcpyHostToDevice, cuda_status, (size_t)alloc_size,
		sizeof(float));

	cuda_status = CopyFunction("cudaMemcpy failed!", v_copy_ptr, v_ptr,
		cudaMemcpyHostToDevice, cuda_status, (size_t)alloc_size,
		sizeof(float3));

	dim3 blocks, threads;
	ThreadAllocator(blocks, threads, length);

	ProjectKernel<<<blocks, threads>>> (result_copy_ptr, curr_copy_ptr, prev_copy_ptr, v_copy_ptr, length, iter, bounds);

	function<cudaError_t()> error_check_func = []() { return cudaGetLastError(); };
	cuda_status = WrapperFunction(error_check_func, "cudaGetLastError (kernel launch)", "LinearSolverKernel", cuda_status);

	function<cudaError_t()> sync_func = []() { return cudaDeviceSynchronize(); };
	cuda_status = WrapperFunction(sync_func, "cudaDeviceSynchronize", "LinearSolverKernel", cuda_status);

	cuda_status = CopyFunction("cudaMemcpy failed!", result_ptr, result_copy_ptr,
		cudaMemcpyDeviceToHost, cuda_status, (size_t)alloc_size,
		sizeof(float3));

	cuda_status = CopyFunction("cudaMemcpy failed!", prev_ptr, prev_copy_ptr,
		cudaMemcpyDeviceToHost, cuda_status, (size_t)alloc_size,
		sizeof(float));

	cuda_status = CopyFunction("cudaMemcpy failed!", current_ptr, curr_copy_ptr,
		cudaMemcpyDeviceToHost, cuda_status, (size_t)alloc_size,
		sizeof(float));

	if (cuda_status != cudaSuccess) {
		CudaMemoryFreer(bidoof);
		CudaMemoryFreer(bidoof2);
	}

	return std::make_tuple(result_ptr, current_ptr, prev_ptr);
}