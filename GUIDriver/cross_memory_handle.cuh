#pragma once

#include <cuda.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <vulkan/vulkan.h>

#include "../CUDATest/handler_methods.hpp"

#ifdef _WIN64
#include <AclAPI.h>
#include <vulkan/vulkan_win32.h>
typedef HANDLE ShareableHandle;
#else
typedef int ShareableHandle;
#endif

class CrossMemoryHandle {
public:
	CrossMemoryHandle() = default;

	CrossMemoryHandle(const size_t& size_in, const size_t& type_size_in, const bool& host_inclusive_in = true);

	VkDeviceSize TotalAllocationSize() const;

	cudaError_t AllocateCudaMemory();

	cudaError_t DeallocateCudaMemory();

	VkBuffer buffer = {};
	VkDeviceMemory buffer_memory = {};

	CUmemGenericAllocationHandle cuda_handle = {};
	ShareableHandle shareable_handle = {};

	void* vulkan_ptr = nullptr;
	void* cuda_host_ptr = nullptr, *cuda_device_ptr = nullptr;

	size_t size = 0;
	size_t granularity_size = 0;
	size_t type_size = 0;
private:
	bool host_inclusive = true;
};