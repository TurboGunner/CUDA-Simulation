#include "cuda_interop_helper.cuh"

#ifdef _WIN64
#include <VersionHelpers.h>
#include <winternl.h>
#include <sddl.h>
#endif

CudaInterop::CudaInterop(VkDevice& device_in, VkPhysicalDevice& phys_device_in) {
    device_ = device_in;
    phys_device_ = phys_device_in;

#ifdef _WIN64
    os_ = IsWindows8OrGreater() ? WINDOWS_MODERN : WINDOWS_OLD;
#else
    os_ = LINUX;
#endif
    if (os_ != LINUX) {
        ipc_handle_type_flag_ = CU_MEM_HANDLE_TYPE_WIN32;
    }
    else {
        ipc_handle_type_flag_ = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    }

    cudaError_t cuda_status = cudaGetDeviceCount(&device_count_);

    if (device_count_ > 1) {
        ProgramLog::OutputLine("Warning: There are multiple CUDA devices!");
    }

    cuda_status = cudaGetDevice(&cuda_device_);
}

void CudaInterop::GetDefaultSecurityDescriptor(CUmemAllocationProp* prop) {
    if (os_ == LINUX) {
        return;
    }
    static const char sddl[] = "D:P(OA;;GARCSDWDWOCCDCLCSWLODTWPRPCRFA;;;WD)";
    static OBJECT_ATTRIBUTES obj_attributes;
    static bool obj_attributes_configured = false;

    if (!obj_attributes_configured) {
        PSECURITY_DESCRIPTOR security_descriptor;
        BOOL result = ConvertStringSecurityDescriptorToSecurityDescriptorA(sddl, SDDL_REVISION_1, &security_descriptor, NULL); //NOTE
        if (result == 0) {
            ProgramLog::OutputLine("IPC failure: GetDefaultSecurityDescriptor Failed! (%d)\n", GetLastError());
        }

        InitializeObjectAttributes(&obj_attributes, nullptr, 0, nullptr, security_descriptor); //NOTE

        obj_attributes_configured = true;
    }
    prop->win32HandleMetaData = &obj_attributes;
}

size_t CudaInterop::RoundWarpGranularity(const size_t& size, const size_t& granularity) {
    return ((size + granularity - 1) / granularity) * granularity;
}

void CudaInterop::CalculateTotalMemorySize(const size_t& granularity) {
    total_alloc_size_ = 0;

    for (auto& mem_handle : cross_memory_handles_) {
        size_t current_granularity_size = RoundWarpGranularity(mem_handle.TotalAllocationSize(), granularity);
        total_alloc_size_ += current_granularity_size;
        mem_handle.granularity_size = current_granularity_size;
    }
}

void CudaInterop::AddMemoryHandle(const size_t& size, const size_t& type_size) {  //NOTE: ALLOCATE STRUCT WITH SIZE
    //WIP

    CrossMemoryHandle position_handle(size, type_size);
    cross_memory_handles_.push_back(position_handle);
}

cudaError_t CudaInterop::CreateStream(const unsigned int& flags) {
    return cudaStreamCreateWithFlags(&cuda_stream_, flags);
}

void CudaInterop::MemoryAllocationProp() {
    current_alloc_prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;

    current_alloc_prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    current_alloc_prop_.location.id = cuda_device_;

    current_alloc_prop_.win32HandleMetaData = nullptr;
    current_alloc_prop_.requestedHandleTypes = ipc_handle_type_flag_;

    if (os_ != LINUX) {
        GetDefaultSecurityDescriptor(&current_alloc_prop_);
    }
}

void CudaInterop::MemoryAccessDescriptor() {
    access_descriptor_ = {};
    access_descriptor_.location.id = cuda_device_;
    access_descriptor_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_descriptor_.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
}

CUresult CudaInterop::SimulationSetupAllocations() {
    CUdeviceptr d_ptr = 0U;
    size_t granularity = 0;

    CUresult cuda_result;

    MemoryAllocationProp();

    cuda_result = cuMemGetAllocationGranularity(&granularity, &current_alloc_prop_, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    DriverLog(cuda_result, "Allocation Granularity");

    CalculateTotalMemorySize(granularity);

    cuda_result = cuMemAddressReserve(&d_ptr, total_alloc_size_, granularity, 0U, 0);
    DriverLog(cuda_result, "MemAddressReserve");

    ProgramLog::OutputLine("Granularity: " + std::to_string(granularity) + "\n");

    cuda_result = cuMemCreate(&cross_memory_handles_[0].cuda_handle, cross_memory_handles_[0].granularity_size, &current_alloc_prop_, 0);
    DriverLog(cuda_result, "MemCreate");

    cuda_result = cuMemExportToShareableHandle((void*)&cross_memory_handles_[0].shareable_handle, cross_memory_handles_[0].cuda_handle, ipc_handle_type_flag_, 0);
    DriverLog(cuda_result, "ExportToShareableHandle");

    CUdeviceptr va_position = d_ptr; //NOTE: When having other pointers, this will adding the offsets in order to properly account for fitting into the contiguous VA range.
    cross_memory_handles_[0].vulkan_ptr = (void*)va_position;

    cuda_result = cuMemMap(va_position, cross_memory_handles_[0].granularity_size, 0, cross_memory_handles_[0].cuda_handle, 0);
    DriverLog(cuda_result, "MapMemory");

    cuda_result = cuMemRelease(cross_memory_handles_[0].cuda_handle);
    DriverLog(cuda_result, "ReleaseMemory");

    MemoryAccessDescriptor();

    cuda_result = cuMemSetAccess(d_ptr, total_alloc_size_, &access_descriptor_, 1); //Adds read-write access to the whole VA range.
    DriverLog(cuda_result, "SetMemoryAccess");

    return cuda_result;
}

CUresult CudaInterop::Clean() {
    CUresult cuda_result;
    for (const auto& mem_handle : cross_memory_handles_) { //Ensures that all allocations are mapped before attempting to unmap memory
        if (!mem_handle.vulkan_ptr) {
            DriverLog(cuda_result, "Clean");
            return cuda_result;
        }
    }

    IPCCloseShareableHandle(cross_memory_handles_[0].shareable_handle);

    cuda_result = cuMemAddressFree((CUdeviceptr) cross_memory_handles_[0].vulkan_ptr, total_alloc_size_);
    DriverLog(cuda_result, "VulkanPtrCUDAFree");

    return cuda_result;
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

    for (const auto& mem_handle : cross_memory_handles_) {
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

    VkDeviceSize alloc_size = cross_memory_handles_[0].TotalAllocationSize();
    auto mem_handle_type = GetPlatformMemoryHandle();
    void* mem_handle = (void*) (uintptr_t) &cross_memory_handles_[0].shareable_handle;

    VkBuffer& buffer = cross_memory_handles_[0].buffer;
    VkDeviceMemory& buffer_memory = cross_memory_handles_[0].buffer_memory;

    vulkan_status = ImportExternalBuffer(mem_handle, mem_handle_type, alloc_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, buffer_memory);

    if (vulkan_status != VK_SUCCESS) {
        ProgramLog::OutputLine("Importing external buffer failed in InitializeCudaInterop!");
    }

    ProgramLog::OutputLine("Buffer Memory Size: " + std::to_string(alloc_size));

    auto mem_semaphore_type = GetPlatformSemaphoreHandle();

    vulkan_status = CreateExternalSemaphore(wait_semaphore, mem_semaphore_type);
    VulkanExceptionHandler(vulkan_status, "Failure creating wait semaphore in InitializeCudaInterop!");
    vulkan_status = CreateExternalSemaphore(signal_semaphore, mem_semaphore_type);
    VulkanExceptionHandler(vulkan_status, "Failure creating signal semaphore in InitializeCudaInterop!");

    cuda_status = ImportCudaExternalSemaphore(cuda_wait_semaphore_, wait_semaphore, mem_semaphore_type);
    CudaExceptionHandler(cuda_status, "ImportCUDAExternalSemaphoreWait");
    cuda_status = ImportCudaExternalSemaphore(cuda_signal_semaphore_, signal_semaphore, mem_semaphore_type);
    CudaExceptionHandler(cuda_status, "ImportCUDAExternalSemaphoreSignal");

    return cuda_status;
}

void CudaInterop::InteropExtensions() {
    interop_extensions_.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);

    interop_extensions_.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
    interop_extensions_.push_back(VK_KHR_EXTERNAL_FENCE_CAPABILITIES_EXTENSION_NAME);

    interop_extensions_.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
}

void CudaInterop::InteropDeviceExtensions() {
    interop_device_extensions_.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
    interop_device_extensions_.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);

    if (os_ != LINUX) {
        interop_device_extensions_.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
        interop_device_extensions_.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
    }
    else {
        interop_device_extensions_.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
        interop_device_extensions_.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
    }
}

bool CudaInterop::IsVkPhysicalDeviceUUID(void* uuid) {
    return !memcmp((void*) vk_device_uuid_, uuid, (size_t) VK_UUID_SIZE);
}

void CudaInterop::PopulateCommandBuffer(VkCommandBuffer& command_buffer) {
    vkCmdDraw(command_buffer, cross_memory_handles_[0].size, 1, 0, 0);
}