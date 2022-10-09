#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "buffer_helpers.hpp"
#include "vulkan_helpers.hpp"

#include "../CUDATest/handler_methods.hpp"

//Logging
#include "../CUDATest/handler_classes.hpp"

#include "windows_security_attributes.hpp"

#include <vulkan/vulkan.h>

#ifdef _WIN64
#include <vulkan/vulkan_win32.h>
typedef HANDLE ShareableHandle;
#else
typedef int ShareableHandle;
#endif

#include <algorithm>
#include <unordered_map>
#include <string>
#include <vector>

using std::unordered_map;
using std::vector;

enum OperatingSystem { WINDOWS_MODERN, WINDOWS_OLD, LINUX };

class CrossMemoryHandle {
public:
	CrossMemoryHandle() = default;

	CrossMemoryHandle(CUmemGenericAllocationHandle cuda_handle_in, ShareableHandle shareable_handle_in, const size_t& size_in, const size_t& type_size_in);

	VkDeviceSize TotalAllocationSize() const;

	cudaError_t AllocateCudaMemory();

	cudaError_t DeallocateCudaMemory();

	VkBuffer buffer;
	VkDeviceMemory buffer_memory;

	CUmemGenericAllocationHandle cuda_handle;
	ShareableHandle shareable_handle;

	void* vulkan_ptr = nullptr;
	void* cuda_host_ptr = nullptr, *cuda_device_ptr = nullptr;

	size_t size = 0;
	size_t granularity_size = 0;
	size_t type_size = 0;
};

__global__ void TestKernel(float* data);

class CudaInterop {
public:
	CudaInterop() = default;

	CudaInterop(VkDevice& device_in, VkPhysicalDevice& phys_device_in);

	VkExternalSemaphoreHandleTypeFlagBits GetPlatformSemaphoreHandle();
	VkExternalMemoryHandleTypeFlagBits GetPlatformMemoryHandle();

	VkResult CreateExternalSemaphore(VkSemaphore& semaphore, const VkExternalSemaphoreHandleTypeFlagBits& handle_type);

	VkExportSemaphoreWin32HandleInfoKHR ExportSemaphoreHandleWin32();

	VkExportSemaphoreCreateInfoKHR& ExportSemaphoreCreationSettings(const VkExternalSemaphoreHandleTypeFlagBits& handle_type);

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

	cudaError_t ImportCudaExternalMemory(void** cuda_ptr, cudaExternalMemory_t& cuda_memory, VkDeviceMemory& buffer_memory, const VkDeviceSize& size, const VkExternalMemoryHandleTypeFlagBits& handle_type);

	void GetDefaultSecurityDescriptor(CUmemAllocationProp* prop);

	size_t RoundWarpGranularity(const size_t& size, const int& granularity);

	void CalculateTotalMemorySize(const size_t& granularity);

	void AddMemoryHandle(const size_t& size, const size_t& type_size);

	cudaError_t CreateStream(const unsigned int& flags = cudaStreamNonBlocking);

	void PopulateCommandBuffer(VkCommandBuffer& command_buffer);

	void MemoryAllocationProp();

	void MemoryAccessDescriptor();

	CUresult SimulationSetupAllocations();

	CUresult Clean();

	cudaError_t CleanSynchronization();

	void InteropDeviceExtensions();

	int IPCCloseShareableHandle(ShareableHandle sh_handle);

	cudaError_t InitializeCudaInterop(VkSemaphore& wait_semaphore, VkSemaphore& signal_semaphore);

	bool IsVkPhysicalDeviceUUID(void* uuid);

	__host__ cudaError_t TestMethod(VkSemaphore& wait_semaphore, VkSemaphore& signal_semaphore);

	__host__ cudaError_t BulkInitializationTest(VkSemaphore& wait_semaphore, VkSemaphore& signal_semaphore);

	__host__ cudaError_t InteropDrawFrame(VkSemaphore& wait_semaphore, VkSemaphore& signal_semaphore);

	VkDevice device_;
	VkPhysicalDevice phys_device_;

	CUmemAllocationHandleType ipc_handle_type_flag_;
	CUmemAllocationProp current_alloc_prop_ = {};
	CUmemAccessDesc access_descriptor_ = {};

	vector<CrossMemoryHandle> cross_memory_handles_;
	size_t total_alloc_size_ = 0;

	cudaStream_t cuda_stream_;
	vector<const char*> interop_device_extensions_;

	cudaExternalSemaphore_t cuda_wait_semaphore_, cuda_signal_semaphore_;

	int cuda_device_ = -1, device_count_ = 0;
	uint8_t vk_device_uuid_[VK_UUID_SIZE];

	OperatingSystem os_;

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
};