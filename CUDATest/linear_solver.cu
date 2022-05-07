#include "fluid_sim_cuda.cuh"
#include "handler_methods.hpp"
#include "fluid_sim.hpp"

__global__ void LinearSolverKernel(float* result_ptr, float* data, const float* data_prev, float a_fac, float c_fac, unsigned int length, unsigned int iter, int bounds) {
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;

	if (threadIdx.x < length - 1 && threadIdx.y < length - 1) {
		for (int i = 0; i < iter; i++) {
			data[IX(x_bounds, y_bounds + 1, length)] = (data_prev[IX(x_bounds, y_bounds + 1, length)] +
				a_fac *
				(data[IX(x_bounds + 1, y_bounds + 1, length)]
					+ data[IX(x_bounds - 1, y_bounds + 1, length)]
					+ data[IX(x_bounds, y_bounds + 2, length)]
					+ data[IX(x_bounds, y_bounds, length)]))
				* (1.0f / c_fac);
		}
		if (bounds == 0) {
			PointerBoundaries(data, length);
		}
		else {
			PointerBoundariesSpecialX(data, length);
		}
		result_ptr[IX(x_bounds, y_bounds + 1, length)] = data[IX(x_bounds, y_bounds + 1, length)];
	}
}

float* LinearSolverCuda(int bounds, VectorField& current, VectorField& previous, const float& a_fac, const float& c_fac, const unsigned int& iter, const unsigned int& length) {
	float* curr_copy_ptr = nullptr, *prev_copy_ptr = nullptr;

	float* current_ptr = current.FlattenMapX(),
		*prev_ptr = previous.FlattenMapX();

	unsigned int alloc_size = length * length;

	float* result_ptr = new float[alloc_size],
		*result_copy_ptr = nullptr;

	vector<reference_wrapper<float*>> bidoof;
	bidoof.insert(bidoof.end(), { curr_copy_ptr, prev_copy_ptr, result_copy_ptr });

	CudaMemoryAllocator(bidoof, (size_t) alloc_size, sizeof(float));

	cudaError_t cuda_status = cudaSuccess;

	cuda_status = CopyFunction("cudaMemcpy failed!", curr_copy_ptr, current_ptr,
		cudaMemcpyHostToDevice, cuda_status, (size_t) alloc_size,
		sizeof(float));

	cuda_status = CopyFunction("cudaMemcpy failed!", prev_copy_ptr, prev_ptr,
		cudaMemcpyHostToDevice, cuda_status, (size_t) alloc_size,
		sizeof(float));

	dim3 blocks, threads;
	ThreadAllocator(blocks, threads, length);

	LinearSolverKernel<<<blocks, threads>>> (result_copy_ptr, curr_copy_ptr, prev_copy_ptr, a_fac, c_fac, length, iter, bounds);

	function<cudaError_t()> error_check_func = []() { return cudaGetLastError(); };
	cuda_status = WrapperFunction(error_check_func, "cudaGetLastError (kernel launch)", "LinearSolverKernel", cuda_status);

	function<cudaError_t()> sync_func = []() { return cudaDeviceSynchronize(); };
	cuda_status = WrapperFunction(sync_func, "cudaDeviceSynchronize", "LinearSolverKernel", cuda_status);

	cuda_status = CopyFunction("cudaMemcpy failed!", result_ptr, result_copy_ptr,
		cudaMemcpyDeviceToHost, cuda_status, (size_t) alloc_size,
		sizeof(float));

	if (cuda_status != cudaSuccess) {
		CudaMemoryFreer(bidoof);
	}

	return result_ptr;
}