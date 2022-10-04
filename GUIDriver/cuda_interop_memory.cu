#include "cuda_interop_helper.cuh"

#ifdef _WIN64
#include <dxgi1_2.h>
#endif

VkExternalMemoryHandleTypeFlagBits CudaInterop::GetPlatformMemoryHandle() {
	return memory_handle_map[os_];
}

VkResult CudaInterop::CreateExternalBuffer(const VkDeviceSize& size, const VkBufferUsageFlags& usage, const VkMemoryPropertyFlags& properties, const VkExternalMemoryHandleTypeFlagsKHR& external_mem_handle_type, VkBuffer& buffer, VkDeviceMemory& buffer_memory) {
	VkResult vulkan_status = VK_SUCCESS;

	VkBufferCreateInfo buffer_info = BufferHelpers::CreateBufferInfo(size, usage);

	VkExternalMemoryBufferCreateInfo external_buffer_info = {};

	external_buffer_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
	external_buffer_info.handleTypes = external_mem_handle_type;

	if (vkCreateBuffer(device_, &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
		ProgramLog::OutputLine("Error: Failed to properly create buffer!");
	}

	VkMemoryRequirements mem_requirements;

	vkGetBufferMemoryRequirements(device_, buffer, &mem_requirements);

	VkExportMemoryAllocateInfoKHR vulkan_export_memory_allocate_info = {};
	vulkan_export_memory_allocate_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;

	ExportMemoryAllocationSettings(vulkan_export_memory_allocate_info, external_mem_handle_type);

	VkMemoryAllocateInfo alloc_info = VulkanHelper::CreateAllocationInfo(phys_device_, mem_requirements, properties, true); //NOTE
	alloc_info.pNext = &vulkan_export_memory_allocate_info;

	if (vkAllocateMemory(device_, &alloc_info, nullptr, &buffer_memory) != VK_SUCCESS) {
		ProgramLog::OutputLine("Error: Failed to allocate buffer memory!");
	}

	vulkan_status = vkBindBufferMemory(device_, buffer, buffer_memory, 0);

	return vulkan_status;
}

VkExportMemoryWin32HandleInfoKHR CudaInterop::ExportMemoryHandleWin32() {
	VkExportMemoryWin32HandleInfoKHR vulkan_export_memory_win32_info = {};

	vulkan_export_memory_win32_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
	vulkan_export_memory_win32_info.pNext = nullptr;

	vulkan_export_memory_win32_info.pAttributes = &win_security_attributes_;
	vulkan_export_memory_win32_info.dwAccess = DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
	vulkan_export_memory_win32_info.name = (LPCWSTR) nullptr;

	return vulkan_export_memory_win32_info;
}

void CudaInterop::ExportMemoryAllocationSettings(VkExportMemoryAllocateInfoKHR& vulkan_export_memory_allocate_info, const VkExternalMemoryHandleTypeFlagsKHR& external_mem_handle_type) {
	VkExportMemoryWin32HandleInfoKHR vulkan_export_memory_win32_info = {};

	if (os_ != LINUX) {
		vulkan_export_memory_win32_info = ExportMemoryHandleWin32();

		vulkan_export_memory_allocate_info.pNext = external_mem_handle_type & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR ? &vulkan_export_memory_win32_info : nullptr;
		vulkan_export_memory_allocate_info.handleTypes = external_mem_handle_type;
		return;
	}

	vulkan_export_memory_allocate_info.pNext = nullptr;
	vulkan_export_memory_allocate_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
}

VkResult CudaInterop::ImportExternalBuffer(void* handle, const VkExternalMemoryHandleTypeFlagBits& handle_type, const VkDeviceSize& size, const VkBufferUsageFlags& usage, const VkMemoryPropertyFlags& properties, VkBuffer& buffer, VkDeviceMemory& buffer_memory) {
	VkResult vulkan_status = VK_SUCCESS;

	VkBufferCreateInfo buffer_info = BufferHelpers::CreateBufferInfo(size, usage);

	VkExternalMemoryBufferCreateInfo external_buffer_memory_info = {};

	external_buffer_memory_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
	external_buffer_memory_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

	buffer_info.pNext = &external_buffer_memory_info;

	if (vkCreateBuffer(device_, &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
		ProgramLog::OutputLine("Error: Failed to properly create buffer!");
	}

	VkMemoryRequirements mem_requirements;

	vkGetBufferMemoryRequirements(device_, buffer, &mem_requirements);

	if (os_ != LINUX) {
		VkImportMemoryWin32HandleInfoKHR handle_info = {};

		handle_info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
		handle_info.pNext = nullptr;
		handle_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
		handle_info.handle = handle;
		handle_info.name = nullptr;
	}
	else {
		VkImportMemoryFdInfoKHR handle_info = {};

		handle_info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
		handle_info.pNext = nullptr;
		handle_info.fd = (int)(uintptr_t)handle;
		handle_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
	}

	VkMemoryAllocateInfo alloc_info = VulkanHelper::CreateAllocationInfo(phys_device_, mem_requirements, properties, true);

	if (vkAllocateMemory(device_, &alloc_info, nullptr, &buffer_memory) != VK_SUCCESS) {
		ProgramLog::OutputLine("Error: Failed to allocate buffer memory!");
	}

	vulkan_status = vkBindBufferMemory(device_, buffer, buffer_memory, 0);

	return vulkan_status;
}

void* CudaInterop::GetMemoryHandle(VkDeviceMemory& memory, const VkExternalMemoryHandleTypeFlagBits& handle_type) {
	if (os_ != LINUX) {
		return GetMemoryHandleWin32(memory, handle_type);
	}
	else {
		return GetMemoryHandlePOSIX(memory, handle_type);
	}
}

void* CudaInterop::GetMemoryHandleWin32(VkDeviceMemory& memory, const VkExternalMemoryHandleTypeFlagBits& handle_type) {
	HANDLE handle = 0;

	VkMemoryGetWin32HandleInfoKHR vk_memory_get_win32_handle_info = {};

	vk_memory_get_win32_handle_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
	vk_memory_get_win32_handle_info.pNext = nullptr;
	vk_memory_get_win32_handle_info.memory = memory;
	vk_memory_get_win32_handle_info.handleType = handle_type;

	PFN_vkGetMemoryWin32HandleKHR func_get_memory_win32_handle = (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device_, "vkGetMemoryWin32HandleKHR");
	if (!func_get_memory_win32_handle) {
		ProgramLog::OutputLine("Error: Failed to retrieve vkGetMemoryWin32HandleKHR!");
	}
	if (func_get_memory_win32_handle(device_, &vk_memory_get_win32_handle_info, &handle) != VK_SUCCESS) {
		ProgramLog::OutputLine("Error: Failed to retrieve handle for buffer!");
	}

	return (void*)handle;
}

void* CudaInterop::GetMemoryHandlePOSIX(VkDeviceMemory& memory, const VkExternalMemoryHandleTypeFlagBits& handle_type) {
	int fd = -1;

	VkMemoryGetFdInfoKHR memory_get_fd_info = {};

	memory_get_fd_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
	memory_get_fd_info.pNext = nullptr;
	memory_get_fd_info.memory = memory;
	memory_get_fd_info.handleType = handle_type;

	PFN_vkGetMemoryFdKHR fpGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR");
	if (!fpGetMemoryFdKHR) {
		ProgramLog::OutputLine("Error: Failed to retrieve vkGetMemoryFdHandleKHR!");
	}
	if (fpGetMemoryFdKHR(device_, &memory_get_fd_info, &fd) != VK_SUCCESS) {
		ProgramLog::OutputLine("Error: Failed to retrieve handle for buffer!");
	}
	return (void*)(uintptr_t) fd;
}

cudaError_t CudaInterop::ImportCudaExternalMemory(void** cuda_ptr, cudaExternalMemory_t& cuda_memory, VkDeviceMemory& buffer_memory, const VkDeviceSize& size, const VkExternalMemoryHandleTypeFlagBits& handle_type) {
	cudaError_t cuda_status = cudaSuccess;
	VkResult vulkan_status = VK_SUCCESS;

	cudaExternalMemoryHandleDesc external_memory_handle_desc = {};

	if (handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
		external_memory_handle_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
	}
	else if (handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
		external_memory_handle_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
	}
	else if (handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
		external_memory_handle_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
	}
	else {
		ProgramLog::OutputLine("Error: Unknown handle type requested!");
	}

	external_memory_handle_desc.size = size;

	if (os_ != LINUX) {
		external_memory_handle_desc.handle.win32.handle = (HANDLE) GetMemoryHandle(buffer_memory, handle_type);
	}
	else {
		external_memory_handle_desc.handle.fd = (int) (uintptr_t) GetMemoryHandle(buffer_memory, handle_type);
	}

	cuda_status = cudaImportExternalMemory(&cuda_memory, &external_memory_handle_desc);

	cudaExternalMemoryBufferDesc external_mem_buffer_desc = {};

	external_mem_buffer_desc.offset = 0;
	external_mem_buffer_desc.size = size;
	external_mem_buffer_desc.flags = 0;

	cuda_status = cudaExternalMemoryGetMappedBuffer(cuda_ptr, cuda_memory, &external_mem_buffer_desc);

	return cuda_status;
}