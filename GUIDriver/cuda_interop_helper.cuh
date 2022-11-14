#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "buffer_helpers.hpp"
#include "vulkan_helpers.hpp"

#include "interop_memory_allocator.cuh"
#include "cross_memory_handle.cuh"

#include "../CUDATest/handler_methods.hpp"

//Logging
#include "../CUDATest/handler_classes.hpp"

#include "windows_security_attributes.hpp"

#include "../Meshing/mpm.cuh"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <unordered_map>
#include <string>
#include <vector>

using std::string;
using std::unordered_map;
using std::vector;

static const unordered_map<OperatingSystem, VkExternalSemaphoreHandleTypeFlagBits> semaphore_handle_map = {
	{ OperatingSystem::WINDOWS_MODERN, VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT },
	{ OperatingSystem::WINDOWS_OLD, VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT },
	{ OperatingSystem::LINUX, VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT }
};

static const unordered_map<OperatingSystem, VkExternalMemoryHandleTypeFlagBits> memory_handle_map = {
	{ OperatingSystem::WINDOWS_MODERN, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT },
	{ OperatingSystem::WINDOWS_OLD, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT },
	{ OperatingSystem::LINUX, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT }
};

class CudaInterop {
public:
	CudaInterop() = default;

	CudaInterop(VkDevice& device_in, VkPhysicalDevice& phys_device_in);

	VkExternalSemaphoreHandleTypeFlagBits GetPlatformSemaphoreHandle();
	VkExternalMemoryHandleTypeFlagBits GetPlatformMemoryHandle();

	VkResult CreateExternalSemaphore(VkSemaphore& semaphore, const VkExternalSemaphoreHandleTypeFlagBits handle_type);

	VkExportSemaphoreWin32HandleInfoKHR ExportSemaphoreHandleWin32();

	VkResult CreateExternalBuffer(const VkDeviceSize size, const VkBufferUsageFlags usage, const VkMemoryPropertyFlags properties, const VkExternalMemoryHandleTypeFlagsKHR external_mem_handle_type, VkBuffer& buffer, VkDeviceMemory& buffer_memory);

	VkExportMemoryWin32HandleInfoKHR ExportMemoryHandleWin32();

	VkResult ImportExternalBuffer(void* handle, const VkExternalMemoryHandleTypeFlagBits handle_type, const VkDeviceSize size, const VkBufferUsageFlags usage, const VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& buffer_memory);

	void* GetMemoryHandle(VkDeviceMemory& memory, const VkExternalMemoryHandleTypeFlagBits handle_type);

	void* GetMemoryHandleWin32(VkDeviceMemory& memory, const VkExternalMemoryHandleTypeFlagBits handle_type);

	void* GetMemoryHandlePOSIX(VkDeviceMemory& memory, const VkExternalMemoryHandleTypeFlagBits handle_type);

	void* GetSemaphoreHandle(VkSemaphore& semaphore, const VkExternalSemaphoreHandleTypeFlagBits handle_type);

	void* GetSemaphoreHandleWin32(VkSemaphore& semaphore, const VkExternalSemaphoreHandleTypeFlagBits handle_type);

	void* GetSemaphoreHandlePOSIX(VkSemaphore& semaphore, const VkExternalSemaphoreHandleTypeFlagBits handle_type);

	cudaError_t ImportCudaExternalSemaphore(cudaExternalSemaphore_t& cuda_semaphore, VkSemaphore& vk_semaphore, const VkExternalSemaphoreHandleTypeFlagBits handle_type);

	cudaError_t ImportCudaExternalMemory(void** cuda_ptr, cudaExternalMemory_t& cuda_memory, VkDeviceMemory& buffer_memory, const VkDeviceSize size, const VkExternalMemoryHandleTypeFlagBits handle_type);

	cudaError_t CreateStream(const unsigned int flags = cudaStreamNonBlocking);

	void PopulateCommandBuffer(VkCommandBuffer& command_buffer);

	CUresult Clean();

	cudaError_t CleanSynchronization();


	int IPCCloseShareableHandle(ShareableHandle sh_handle);

	cudaError_t InitializeCudaInterop(VkSemaphore& wait_semaphore, VkSemaphore& signal_semaphore);

	bool IsVkPhysicalDeviceUUID(void* uuid);

	__host__ cudaError_t TestMethod();

	__host__ cudaError_t BulkInitializationTest(VkSemaphore& wait_semaphore, VkSemaphore& signal_semaphore, const size_t size);

	__host__ cudaError_t InteropDrawFrame(VkSemaphore& wait_semaphore, VkSemaphore& signal_semaphore);

	vector<const char*> interop_extensions_ = {
		VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
		VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
		VK_KHR_EXTERNAL_FENCE_CAPABILITIES_EXTENSION_NAME,
		VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
	};

	vector<const char*> interop_device_extensions_ = {
		VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
		VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
#ifdef _WIN64
		VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
		VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME
#else
		VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
		VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME
#endif
	};

	cudaExternalSemaphore_t cuda_wait_semaphore_ = {}, cuda_signal_semaphore_ = {};
	Grid* grid_ = nullptr;

private:
	VkDevice device_;
	VkPhysicalDevice phys_device_;

	size_t total_alloc_size_ = 0;

	cudaStream_t cuda_stream_;

	int cuda_device_ = -1, device_count_ = 0;
	uint8_t vk_device_uuid_[VK_UUID_SIZE];

	OperatingSystem os_ = OperatingSystem::WINDOWS_MODERN;
};