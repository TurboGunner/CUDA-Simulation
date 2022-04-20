#include "fluid_sim_cuda.cuh"
#include "handler_methods.hpp"

#include <iostream>
#include <functional>
#include <vector>

#include <math.h>

using std::reference_wrapper;
using std::vector;

using std::map;

__global__ void LinearSolverKernel(float* result_ptr, const float* data, const float* data_prev, float a_fac, float c_fac, unsigned int length) {
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;

	if (threadIdx.y < length && threadIdx.x < length && x_bounds * y_bounds < 15) {
		result_ptr[IX(x_bounds, y_bounds, length)] = (data_prev[IX(x_bounds - 1, y_bounds, length)] +
			a_fac *
			(data[IX(x_bounds - 1, y_bounds, length)]
			+ data[IX(x_bounds + 1, y_bounds, length)]
			+ data[IX(x_bounds, y_bounds - 1, length)]
			+ data[IX(x_bounds, y_bounds + 1, length)]))
			* (1.0f / c_fac);
	}
}

__device__ int IX(unsigned int x, unsigned int y, const unsigned int& size) {
	unsigned int value = (((y - 1) * size) + x);
	if (value < (size * size)) {
		return value;
	}
	printf("Error: Index Value %d was out of bounds!\n", value);
}

float* LinearSolverCuda(int bounds, VectorField& current, VectorField& previous, const float& a_fac, const float& c_fac) {
	float* curr_copy_ptr = nullptr, *prev_copy_ptr = nullptr;

	float* current_ptr = current.FlattenMapX(),
		*prev_ptr = previous.FlattenMapX();

	float* result_ptr = new float[current.GetVectorMap().size()],
		*result_copy_ptr = nullptr;

	vector<reference_wrapper<float*>> bidoof;
	bidoof.insert(bidoof.end(), { curr_copy_ptr, prev_copy_ptr, result_copy_ptr });

	CudaMemoryAllocator(bidoof, (size_t) current.GetVectorMap().size(), sizeof(float));

	std::cout << sizeof(current_ptr) * sizeof(float) * 4 << std::endl;

	cudaError_t cuda_status = cudaSuccess;

	cuda_status = CopyFunction("cudaMemcpy failed!", curr_copy_ptr, current.FlattenMapX(),
		cudaMemcpyHostToDevice, cuda_status, (size_t) current.GetVectorMap().size(),
		sizeof(float));

	cuda_status = CopyFunction("cudaMemcpy failed!", prev_copy_ptr, previous.FlattenMapX(),
		cudaMemcpyHostToDevice, cuda_status, (size_t) current.GetVectorMap().size(),
		sizeof(float));

	int thread_count = 16;
	int block_count = ((current.GetVectorMap().size() + thread_count) - 1) / thread_count;

	dim3 threads(thread_count, thread_count);
	dim3 blocks(block_count, block_count);

	LinearSolverKernel<<<blocks, threads>>> (result_copy_ptr, curr_copy_ptr, prev_copy_ptr, a_fac, c_fac, (int)sqrt(current.GetVectorMap().size()));

	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cuda_status));
		CudaMemoryFreer(bidoof);
		return result_ptr;
	}

	cuda_status = cudaDeviceSynchronize();

	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching LinearSolverKernel!\n", cuda_status);
		printf(cudaGetErrorString(cuda_status));
		printf("\n");
		CudaMemoryFreer(bidoof);
		return result_ptr;
	}

	cuda_status = CopyFunction("cudaMemcpy failed!", result_ptr, result_copy_ptr,
		cudaMemcpyDeviceToHost, cuda_status, (size_t)current.GetVectorMap().size(),
		sizeof(float));


	if (cuda_status != cudaSuccess) {
		std::cout << "Not working!" << std::endl;
		CudaMemoryFreer(bidoof);
	}

	return result_ptr;
}