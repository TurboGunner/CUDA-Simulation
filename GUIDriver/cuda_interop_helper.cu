#include "cuda_interop_helper.cuh"

#ifdef _WIN64
#include <VersionHelpers.h>
#include <winternl.h>
#include <sddl.h>
#endif

CudaInterop::CudaInterop(VkDevice& device_in, VkPhysicalDevice& phys_device_in) {
    device_ = device_in;
    phys_device_ = phys_device_in;
}

cudaError_t CudaInterop::CreateStream(const unsigned int flags) {
    return cudaStreamCreateWithFlags(&cuda_stream_, flags);
}

CUresult CudaInterop::Clean() {
    return InteropMemoryHandler::Clean();
}

cudaError_t CudaInterop::CleanSynchronization() {
    cudaError_t cuda_status = cudaSuccess;

    if (cuda_stream_) {
        cuda_status = cudaStreamSynchronize(cuda_stream_);
        cuda_status = cudaStreamDestroy(cuda_stream_);
    }

    cuda_status = cudaDestroyExternalSemaphore(cuda_wait_semaphore_);
    CudaExceptionHandler(cuda_status, "DestroyExternalSemaphoreWait");
    cuda_status = cudaDestroyExternalSemaphore(cuda_signal_semaphore_);
    CudaExceptionHandler(cuda_status, "DestroyExternalSemaphoreSignal");

    for (const auto& mem_handle : InteropMemoryHandler::CrossMemoryHandles()) {
        vkDestroyBuffer(device_, mem_handle.buffer, nullptr);
        vkFreeMemory(device_, mem_handle.buffer_memory, nullptr);
    }

    return cuda_status;
}

int CudaInterop::IPCCloseShareableHandle(ShareableHandle sh_handle) {
    return CloseHandle(sh_handle);
}

cudaError_t CudaInterop::InitializeCudaInterop(VkSemaphore& wait_semaphore, VkSemaphore& signal_semaphore) {
    cudaError_t cuda_status = cudaSuccess;
    VkResult vulkan_status = VK_SUCCESS;

    VkDeviceSize alloc_size = InteropMemoryHandler::CrossMemoryHandles()[0].TotalAllocationSize();
    auto mem_handle_type = GetPlatformMemoryHandle();
    void* mem_handle = (void*) (uintptr_t)InteropMemoryHandler::CrossMemoryHandles()[0].shareable_handle;

    vulkan_status = ImportExternalBuffer(mem_handle, mem_handle_type, alloc_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, InteropMemoryHandler::CrossMemoryHandles()[0].buffer, InteropMemoryHandler::CrossMemoryHandles()[0].buffer_memory);

    if (vulkan_status != VK_SUCCESS) {
        ProgramLog::OutputLine("Importing external buffer failed in InitializeCudaInterop!");
    }

    ProgramLog::OutputLine("Buffer Memory Size: " + std::to_string(alloc_size));

    auto mem_semaphore_type = GetPlatformSemaphoreHandle();

    ProgramLog::OutputLine("External Semaphore Enum: " + std::to_string(mem_semaphore_type));

    vulkan_status = CreateExternalSemaphore(wait_semaphore, mem_semaphore_type);
    VulkanExceptionHandler(vulkan_status, "Failure creating wait semaphore in InitializeCudaInterop!");
    vulkan_status = CreateExternalSemaphore(signal_semaphore, mem_semaphore_type);
    VulkanExceptionHandler(vulkan_status, "Failure creating signal semaphore in InitializeCudaInterop!");

    cuda_status = ImportCudaExternalSemaphore(cuda_wait_semaphore_, signal_semaphore, mem_semaphore_type);
    CudaExceptionHandler(cuda_status, "ImportCUDAExternalSemaphoreWait");
    cuda_status = ImportCudaExternalSemaphore(cuda_signal_semaphore_, wait_semaphore, mem_semaphore_type);
    CudaExceptionHandler(cuda_status, "ImportCUDAExternalSemaphoreSignal");

    return cuda_status;
}

bool CudaInterop::IsVkPhysicalDeviceUUID(void* uuid) {
    return !memcmp((void*) vk_device_uuid_, uuid, (size_t) VK_UUID_SIZE);
}

void CudaInterop::PopulateCommandBuffer(VkCommandBuffer& command_buffer) {
    vkCmdDraw(command_buffer, InteropMemoryHandler::CrossMemoryHandles()[0].size, 1, 0, 0);
}