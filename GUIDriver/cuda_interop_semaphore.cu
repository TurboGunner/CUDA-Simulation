#include "cuda_interop_helper.cuh"

VkExternalSemaphoreHandleTypeFlagBits CudaInterop::GetPlatformSemaphoreHandle() {
	return semaphore_handle_map[os_];
}

VkResult CudaInterop::CreateExternalSemaphore(VkSemaphore& semaphore, const VkExternalSemaphoreHandleTypeFlagBits& handle_type) {
	VkResult vulkan_status = VK_SUCCESS;

	VkSemaphoreCreateInfo semaphore_info = {};
	semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkExportSemaphoreCreateInfoKHR export_semaphore_create_info = {};
	export_semaphore_create_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;

	ExportSemaphoreCreationSettings(export_semaphore_create_info, handle_type);

	semaphore_info.pNext = &export_semaphore_create_info;

	if (vkCreateSemaphore(device_, &semaphore_info, nullptr, &semaphore) != VK_SUCCESS) {
		ProgramLog::OutputLine("Failed to create synchronization objects for CUDA-Vulkan interop!");
	}
	return vulkan_status;
}

VkExportSemaphoreWin32HandleInfoKHR CudaInterop::ExportSemaphoreHandleWin32() {
	VkExportSemaphoreWin32HandleInfoKHR export_semaphore_win32_handle = {};

	export_semaphore_win32_handle.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR;
	export_semaphore_win32_handle.pNext = nullptr;

	export_semaphore_win32_handle.pAttributes = &win_security_attributes_;
	export_semaphore_win32_handle.dwAccess = DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;

	export_semaphore_win32_handle.name = (LPCWSTR) nullptr;

	return export_semaphore_win32_handle;
}

void CudaInterop::ExportSemaphoreCreationSettings(VkExportSemaphoreCreateInfoKHR& export_semaphore_create_info, const VkExternalSemaphoreHandleTypeFlagBits& handle_type) {
	VkExportSemaphoreWin32HandleInfoKHR export_semaphore_win32_handle = {};

	if (os_ != LINUX) {
		export_semaphore_win32_handle = ExportSemaphoreHandleWin32();
		export_semaphore_create_info.pNext = (handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) ? &export_semaphore_win32_handle : nullptr;
	}
	else {
		export_semaphore_create_info.pNext = nullptr;
	}
	export_semaphore_create_info.handleTypes = handle_type;
}

void* CudaInterop::GetSemaphoreHandle(VkSemaphore& semaphore, const VkExternalSemaphoreHandleTypeFlagBits& handle_type) {
	if (os_ != LINUX) {
		return GetSemaphoreHandleWin32(semaphore, handle_type);
	}
	else {
		return GetSemaphoreHandlePOSIX(semaphore, handle_type);
	}
}

void* CudaInterop::GetSemaphoreHandleWin32(VkSemaphore& semaphore, const VkExternalSemaphoreHandleTypeFlagBits& handle_type) {
	HANDLE handle;

	VkSemaphoreGetWin32HandleInfoKHR semaphore_get_win32_handle_info = {};

	semaphore_get_win32_handle_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
	semaphore_get_win32_handle_info.pNext = nullptr;
	semaphore_get_win32_handle_info.semaphore = semaphore;
	semaphore_get_win32_handle_info.handleType = handle_type;

	PFN_vkGetSemaphoreWin32HandleKHR func_get_semaphore_win32_handle = (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(device_, "vkGetSemaphoreWin32HandleKHR");
	if (!func_get_semaphore_win32_handle) {
		ProgramLog::OutputLine("Error: Failed to retrieve vkGetSemaphoreWin32HandleKHR!");
	}
	if (func_get_semaphore_win32_handle(device_, &semaphore_get_win32_handle_info, &handle) != VK_SUCCESS) {
		ProgramLog::OutputLine("Error: Failed to retrieve handle for semaphore!");
	}

	return (void*)handle;
}

void* CudaInterop::GetSemaphoreHandlePOSIX(VkSemaphore& semaphore, const VkExternalSemaphoreHandleTypeFlagBits& handle_type) {
	int fd;

	VkSemaphoreGetFdInfoKHR semaphore_get_fd_info = {};

	semaphore_get_fd_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
	semaphore_get_fd_info.pNext = nullptr;
	semaphore_get_fd_info.semaphore = semaphore;
	semaphore_get_fd_info.handleType = handle_type;

	PFN_vkGetSemaphoreFdKHR func_get_semaphore_fd = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(device_, "vkGetSemaphoreFdKHR");
	if (!func_get_semaphore_fd) {
		ProgramLog::OutputLine("Error: Failed to retrieve vkGetSemaphoreFdHandleKHR!");
	}
	if (func_get_semaphore_fd(device_, &semaphore_get_fd_info, &fd) != VK_SUCCESS) {
		ProgramLog::OutputLine("Error: Failed to retrieve handle for semaphore!");
	}

	return (void*)(uintptr_t)fd;
}

cudaError_t CudaInterop::ImportCudaExternalSemaphore(cudaExternalSemaphore_t& cuda_semaphore, VkSemaphore& vk_semaphore, const VkExternalSemaphoreHandleTypeFlagBits& handle_type) {
	cudaError_t cuda_status = cudaSuccess;

	cudaExternalSemaphoreHandleDesc external_semaphore_handle_desc = {};

	if (handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
		external_semaphore_handle_desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
	}
	else if (handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
		external_semaphore_handle_desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
	}
	else if (handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
		external_semaphore_handle_desc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
	}
	else {
		ProgramLog::OutputLine("Error: Unknown handle type requested!");
	}

	if (os_ != LINUX) {
		external_semaphore_handle_desc.handle.win32.handle = (HANDLE) GetSemaphoreHandle(vk_semaphore, handle_type);
	}
	else {
		external_semaphore_handle_desc.handle.fd = (int) (uintptr_t) GetSemaphoreHandle(vk_semaphore, handle_type);
	}

	external_semaphore_handle_desc.flags = 0;

	cuda_status = cudaImportExternalSemaphore(&cuda_semaphore, &external_semaphore_handle_desc);

	return cuda_status;
}