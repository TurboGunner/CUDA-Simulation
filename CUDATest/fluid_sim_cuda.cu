#include "fluid_sim_cuda.cuh"
#include "handler_methods.hpp"
#include "fluid_sim.hpp"

#include <iostream>
#include <functional>
#include <vector>

using std::reference_wrapper;
using std::vector;

using std::map;

__global__ void LinearSolverKernel(float* result_ptr, float* data, const float* data_prev, float a_fac, float c_fac, unsigned int length, unsigned int iter) {
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;

	if (threadIdx.x < length && threadIdx.y < length - 1) {
		for (int i = 0; i < iter; i++) {
			data[IX(x_bounds, y_bounds, length)] = (data_prev[IX(x_bounds, y_bounds, length)] +
				a_fac *
				(data[IX(x_bounds + 1, y_bounds, length)]
					+ data[IX(x_bounds - 1, y_bounds, length)]
					+ data[IX(x_bounds, y_bounds + 1, length)]
					+ data[IX(x_bounds, y_bounds - 1, length)]))
				* (1.0f / c_fac);
		}
		PointerBoundaries(data, length);
		result_ptr[IX(x_bounds, y_bounds, length)] = data[IX(x_bounds, y_bounds, length)];
	}
}

__device__ int IX(unsigned int x, unsigned int y, const unsigned int& size) {
	unsigned int value = (((y - 1) * size) + x);
	if (value < (size * size)) {
		return value;
	}
}

__device__ void PointerBoundaries(float* result_ptr, const unsigned int& length) {
	unsigned int bound = length - 1;
	result_ptr[IX(0, 1, length)] = result_ptr[IX(1, 1, length)] + result_ptr[IX(0, 2, length)] * .5f;
	result_ptr[IX(0, length, length)] = result_ptr[IX(1, length, length)] + result_ptr[IX(0, bound, length)] * .5f;
	result_ptr[IX(bound, 1, length)] = result_ptr[IX(bound - 1, 1, length)] + result_ptr[IX(bound, 2, length)] * .5f;
	result_ptr[IX(bound, length, length)] = result_ptr[IX(bound - 1, length, length)] + result_ptr[IX(bound, bound, length)] * .5f;
}

float* LinearSolverCuda(int bounds, VectorField& current, VectorField& previous, const float& a_fac, const float& c_fac, const unsigned int& iter, const unsigned int& length) {
	float* curr_copy_ptr = nullptr, *prev_copy_ptr = nullptr;

	float* current_ptr = nullptr,
		*prev_ptr = nullptr;

	current_ptr = current.FlattenMapX();
	prev_ptr = previous.FlattenMapX();

	float* result_ptr = new float[length * length],
		*result_copy_ptr = nullptr;

	vector<reference_wrapper<float*>> bidoof;
	bidoof.insert(bidoof.end(), { curr_copy_ptr, prev_copy_ptr, result_copy_ptr });

	CudaMemoryAllocator(bidoof, (size_t) length * length, sizeof(float));

	cudaError_t cuda_status = cudaSuccess;

	cuda_status = CopyFunction("cudaMemcpy failed!", curr_copy_ptr, current_ptr,
		cudaMemcpyHostToDevice, cuda_status, (size_t) length * length,
		sizeof(float));

	cuda_status = CopyFunction("cudaMemcpy failed!", prev_copy_ptr, prev_ptr,
		cudaMemcpyHostToDevice, cuda_status, (size_t) length * length,
		sizeof(float));

	int thread_count = 16;
	int block_count_field = (((length * length) + thread_count) - 1) / thread_count;

	dim3 threads(thread_count, thread_count);
	dim3 blocks(block_count_field, block_count_field);

	LinearSolverKernel<<<blocks, threads>>> (result_copy_ptr, curr_copy_ptr, prev_copy_ptr, a_fac, c_fac, length, iter);

	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cuda_status));
		CudaMemoryFreer(bidoof);
		return result_ptr;
	}

	cuda_status = SyncFunction("LinearSolverKernel", cuda_status);

	cuda_status = CopyFunction("cudaMemcpy failed!", result_ptr, result_copy_ptr,
		cudaMemcpyDeviceToHost, cuda_status, (size_t)current.GetVectorMap().size(),
		sizeof(float));


	if (cuda_status != cudaSuccess) {
		std::cout << "Not working!" << std::endl;
		CudaMemoryFreer(bidoof);
	}

	return result_ptr;
}