#include "cuda_interop_helper.cuh"

__global__ void TestKernel(float* data) {
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	data[x_bounds] = x_bounds;
}

__host__ cudaError_t CudaInterop::TestMethod(VkSemaphore& wait_semaphore, VkSemaphore& signal_semaphore) {
	cudaError_t cuda_status = BulkInitializationTest(wait_semaphore, signal_semaphore);

	void*& device_ptr = cross_memory_handles_[0].cuda_device_ptr,
		*& host_ptr = cross_memory_handles_[0].cuda_host_ptr;
	cuda_status = cudaMemsetAsync(device_ptr, 0, cross_memory_handles_[0].TotalAllocationSize(), cuda_stream_);

	dim3 blocks, threads;

	TestKernel<<<blocks, threads, 0, cuda_stream_>>> ((float*) device_ptr);

	cuda_status = PostExecutionChecks(cuda_status, "TestKernel");

	cuda_status = cudaMemcpyAsync(host_ptr, device_ptr, cross_memory_handles_[0].TotalAllocationSize(), cudaMemcpyDeviceToHost, cuda_stream_);

	return cuda_status;
}

__host__ cudaError_t CudaInterop::BulkInitializationTest(VkSemaphore& wait_semaphore, VkSemaphore& signal_semaphore) {
	cudaError_t cuda_status = cudaSuccess;

	size_t size = 4; //WIP, Test

	AddMemoryHandle(size, sizeof(float)); //Adds memory handle struct

	cuda_status = CreateStream();
	CUresult cuda_result = SimulationSetupAllocations(); //Setups the allocation for the simulation

	for (auto& cross_memory_handle : cross_memory_handles_) {
		cuda_status = cross_memory_handle.AllocateCudaMemory(); //Allocates CUDA memory across handle structs
	}

	cuda_status = InitializeCudaInterop(wait_semaphore, signal_semaphore);

	return cuda_status;
}