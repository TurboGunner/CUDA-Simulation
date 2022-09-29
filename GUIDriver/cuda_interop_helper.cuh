#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "buffer_helpers.hpp"
#include "vulkan_helpers.hpp"

//Logging
#include "../CUDATest/handler_classes.hpp"

#include "windows_security_attributes.h"

#include <vulkan/vulkan.h>

#ifdef _WIN64
#include <aclapi.h>
#include <dxgi1_2.h>
#include <ntdef.h>
#include <sddl.h>
#include <VersionHelpers.h>
#include <vulkan/vulkan_win32.h>
#include <windows.h>
#include <winternl.h>

typedef HANDLE ShareableHandle;
#else
typedef int ShareableHandle;
#endif

#include <algorithm>
#include <unordered_map>
#include <vector>

using std::unordered_map;
using std::vector;

enum OperatingSystem { WINDOWS_MODERN, WINDOWS_OLD, LINUX };

unordered_map<OperatingSystem, VkExternalSemaphoreHandleTypeFlagBits> semaphore_handle_map = {
	{ WINDOWS_MODERN, VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT },
	{ WINDOWS_OLD, VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT },
	{ LINUX, VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT }
};

unordered_map<OperatingSystem, VkExternalMemoryHandleTypeFlagBits> memory_handle_map = {
	{ WINDOWS_MODERN, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT },
	{ WINDOWS_OLD, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT },
	{ LINUX, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT }
};

struct CrossMemoryHandle {
	CUmemGenericAllocationHandle cuda_handle_;
	ShareableHandle shareable_handle_;

	size_t size_;
	size_t granularity_size_;
	void* vulkan_ptr = nullptr;
};

class CudaInterop {
public:
	CudaInterop(VkDevice& device_in, VkPhysicalDevice& phys_device_in);

	VkExternalSemaphoreHandleTypeFlagBits GetPlatformSemaphoreHandle();
	VkExternalMemoryHandleTypeFlagBits GetPlatformMemoryHandle();

	VkResult CreateExternalSemaphore(VkSemaphore& semaphore, const VkExternalSemaphoreHandleTypeFlagBits& handle_type);

	VkExportSemaphoreWin32HandleInfoKHR ExportSemaphoreHandleWin32();

	void ExportSemaphoreCreationSettings(VkExportSemaphoreCreateInfoKHR& export_semaphore_create_info, const VkExternalSemaphoreHandleTypeFlagBits& handle_type);

	VkResult CreateExternalBuffer(const VkDeviceSize& size, const VkBufferUsageFlags& usage, const VkMemoryPropertyFlags& properties, const VkExternalMemoryHandleTypeFlagsKHR& external_mem_handle_type, VkBuffer& buffer, VkDeviceMemory& buffer_memory);

	VkExportMemoryWin32HandleInfoKHR ExportMemoryHandleWin32();

	void ExportMemoryAllocationSettings(VkExportMemoryAllocateInfoKHR& vulkan_export_memory_allocate_info, const VkExternalMemoryHandleTypeFlagsKHR& external_mem_handle_type);

	VkResult ImportExternalBuffer(void* handle, const VkExternalMemoryHandleTypeFlagBits& handle_type, const VkDeviceSize& size, const VkBufferUsageFlags& usage, const VkMemoryPropertyFlags& properties, VkBuffer& buffer, VkDeviceMemory& buffer_memory);

	void* GetMemoryHandle(VkDeviceMemory& memory, const VkExternalMemoryHandleTypeFlagBits& handle_type);

	void* GetMemoryHandleWin32(VkDeviceMemory& memory, const VkExternalMemoryHandleTypeFlagBits& handle_type);

	void* GetMemoryHandlePOSIX(VkDeviceMemory& memory, const VkExternalMemoryHandleTypeFlagBits& handle_type);

	void* GetSemaphoreHandle(VkSemaphore& semaphore, const VkExternalSemaphoreHandleTypeFlagBits& handle_type);

	void* GetSemaphoreHandleWin32(VkSemaphore& semaphore, const VkExternalSemaphoreHandleTypeFlagBits& handle_type);

	void* GetSemaphoreHandlePOSIX(VkSemaphore& semaphore, const VkExternalSemaphoreHandleTypeFlagBits& handle_type);

	cudaError_t ImportCudaExternalSemaphore(cudaExternalSemaphore_t& cuda_semaphore, VkSemaphore& vk_semaphore, const VkExternalSemaphoreHandleTypeFlagBits& handle_type);

	void GetDefaultSecurityDescriptor(CUmemAllocationProp* prop);

	size_t RoundWarpGranularity(const size_t& size, const int& granularity);

	void CalculateTotalMemorySize(const size_t& granularity);

	void AddMemoryHandle(const size_t& size);

	cudaError_t CreateStream(const unsigned int& flags = cudaStreamNonBlocking);

	void MemoryAllocationProp();

	void MemoryAccessDescriptor();

	cudaError_t SimulationSetup();

	CUresult Clean();

	int IPCCloseShareableHandle(ShareableHandle sh_handle);

	VkDevice device_;
	VkPhysicalDevice phys_device_;

	CUmemAllocationHandleType ipc_handle_type_flag_;
	CUmemAllocationProp current_alloc_prop_ = {};
	CUmemAccessDesc access_descriptor_ = {};

	WindowsSecurityAttributes win_security_attributes_;

	vector<CrossMemoryHandle> cross_memory_handles_;
	size_t total_alloc_size_ = 0;

	cudaStream_t cuda_stream_;

	int cuda_device_ = -1;

	OperatingSystem os_;
};