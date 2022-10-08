#include "cuda_interop_helper.cuh"

CrossMemoryHandle::CrossMemoryHandle(CUmemGenericAllocationHandle cuda_handle_in, ShareableHandle shareable_handle_in, const size_t& size_in, const size_t& type_size_in) {
	cuda_handle = cuda_handle_in;
	shareable_handle = shareable_handle_in;

	size = size_in;
	type_size = type_size_in;
}

VkDeviceSize CrossMemoryHandle::TotalAllocationSize() const {
	return size * type_size;
}

cudaError_t CrossMemoryHandle::AllocateCudaMemory() {
	cudaError_t cuda_status = cudaMallocHost(&cuda_host_ptr, TotalAllocationSize());
	cuda_status = cudaMalloc(&cuda_device_ptr, TotalAllocationSize());

	return cuda_status;
}

cudaError_t CrossMemoryHandle::DeallocateCudaMemory() {
	cudaError_t cuda_status = cudaFreeHost(cuda_host_ptr);
	cuda_status = cudaFree(cuda_device_ptr);

	return cuda_status;
}