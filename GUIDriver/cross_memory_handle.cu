#include "cross_memory_handle.cuh"

CrossMemoryHandle::CrossMemoryHandle(const size_t& size_in, const size_t& type_size_in, const bool& host_inclusive_in) {
	size = size_in;
	type_size = type_size_in;
	host_inclusive = host_inclusive_in;
}

VkDeviceSize CrossMemoryHandle::TotalAllocationSize() const {
	return size * type_size;
}

cudaError_t CrossMemoryHandle::AllocateCudaMemory() {
	cudaError_t cuda_status = cudaMalloc(&cuda_device_ptr, TotalAllocationSize());
	if (host_inclusive) {
		cuda_status = cudaMallocHost(&cuda_host_ptr, TotalAllocationSize()); //Allocates page-locked memory
	}

	return cuda_status;
}

cudaError_t CrossMemoryHandle::DeallocateCudaMemory() {
	cudaError_t cuda_status = cudaFree(cuda_device_ptr);
	CudaExceptionHandler(cuda_status, "FreeCUDADevice");
	if (host_inclusive) {
		cuda_status = cudaFreeHost(cuda_host_ptr); //Uses cudaFreeHost, as it is page-locked memory
		CudaExceptionHandler(cuda_status, "FreeCUDAHostPageLocked");
	}

	return cuda_status;
}