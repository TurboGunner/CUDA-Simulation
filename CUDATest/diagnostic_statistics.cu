#include "diagnostic_statistics.cuh"

#include <iostream>

__global__ void Maximum(HashMap* data, uint3 length, float* max) {
	unsigned int z_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int x_bounds = blockIdx.z * blockDim.z + threadIdx.z;

	IndexPair incident(x_bounds, y_bounds, z_bounds);

	if (data->Get(incident.IX(length.x)) >= *max) {
		*max = data->Get(incident.IX(length.x));
	}
}

float MaximumCuda(AxisData& map, const uint3& length) {
	cudaError_t cuda_status = cudaSuccess;

	float* max = nullptr,
		*max_copy = new float();

	dim3 blocks, threads;
	ThreadAllocator(blocks, threads, length.x);

	cudaMalloc(&max, sizeof(float));
	cuda_status = CopyFunction("MaximumCudaDevice", max, max_copy, cudaMemcpyHostToDevice, cuda_status, sizeof(float), 1);

	Maximum<<<blocks, threads>>> (map.map_->device_alloc_, length, max);

	cuda_status = CopyFunction("MaximumCudaDevice", max_copy, max, cudaMemcpyDeviceToHost, cuda_status, sizeof(float), 1);

	cuda_status = PostExecutionChecks(cuda_status, "MaximumCudaKernel");

	float result = *max_copy;

	delete max_copy;
	cudaFree(max);

	return result;
}

__global__ void Total(HashMap* data, uint3 length, float* total) {
	unsigned int z_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int x_bounds = blockIdx.z * blockDim.z + threadIdx.z;

	IndexPair incident(x_bounds, y_bounds, z_bounds);
	*total = *total + data->Get(incident.IX(length.x));
}

float TotalCuda(AxisData& map, const uint3& length) {
	cudaError_t cuda_status = cudaSuccess;

	float* total = nullptr,
		* total_copy = new float();

	dim3 blocks, threads;
	ThreadAllocator(blocks, threads, length.x);

	cudaMalloc(&total, sizeof(float));
	cuda_status = CopyFunction("TotalCudaDevice", total, total_copy, cudaMemcpyHostToDevice, cuda_status, sizeof(float), 1);

	Total<<<blocks, threads>>> (map.map_->device_alloc_, length, total);

	cuda_status = CopyFunction("TotalCudaDevice", total_copy, total, cudaMemcpyDeviceToHost, cuda_status, sizeof(float), 1);

	cuda_status = PostExecutionChecks(cuda_status, "TotalCudaKernel");

	std::cout << *total_copy << std::endl;

	float result = *total_copy;

	delete total_copy;
	cudaFree(total);

	return result;
}