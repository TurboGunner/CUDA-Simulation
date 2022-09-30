#include "cuda_interop_helper.cuh"

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

    cudaError_t cuda_status = cudaGetDevice(&cuda_device_);
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
    return;
}

size_t CudaInterop::RoundWarpGranularity(const size_t& size, const int& granularity) {
    return ((size + granularity - 1) / granularity) * granularity;
}

void CudaInterop::CalculateTotalMemorySize(const size_t& granularity) {
    total_alloc_size_ = 0;

    for (auto& mem_handle : cross_memory_handles_) {
        size_t current_granularity_size = RoundWarpGranularity(mem_handle.size_, granularity);
        total_alloc_size_ += current_granularity_size;
        mem_handle.granularity_size_ = current_granularity_size;
    }
}

void CudaInterop::AddMemoryHandle(const size_t& size) {
    CUmemGenericAllocationHandle cuda_position_handle;
    ShareableHandle position_shareable_handle;
    //WIP
    CrossMemoryHandle position_handle = { cuda_position_handle, position_shareable_handle, size };
    cross_memory_handles_.push_back(position_handle);
}

cudaError_t CudaInterop::CreateStream(const unsigned int& flags) {
    return cudaStreamCreateWithFlags(&cuda_stream_, flags);
}

void CudaInterop::MemoryAllocationProp() {
    current_alloc_prop_ = {};
    current_alloc_prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;

    current_alloc_prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    current_alloc_prop_.location.id = cuda_device_;

    current_alloc_prop_.win32HandleMetaData = nullptr;
    current_alloc_prop_.requestedHandleTypes = ipc_handle_type_flag_;
}

void CudaInterop::MemoryAccessDescriptor() {
    access_descriptor_ = {};
    access_descriptor_.location.id = cuda_device_;
    access_descriptor_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_descriptor_.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
}

cudaError_t CudaInterop::SimulationSetup() {
    CUdeviceptr d_ptr = 0U;
    size_t granularity = 0;

    cudaError_t cuda_status = cudaSuccess;
    CUresult cuda_result;

    MemoryAllocationProp();

    GetDefaultSecurityDescriptor(&current_alloc_prop_);

    cuda_result = cuMemGetAllocationGranularity(&granularity, &current_alloc_prop_, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);

    CalculateTotalMemorySize(granularity);

    cuda_result = cuMemAddressReserve(&d_ptr, total_alloc_size_, granularity, 0U, 0);

    cuda_result = cuMemCreate(&cross_memory_handles_[0].cuda_handle_, cross_memory_handles_[0].granularity_size_, &current_alloc_prop_, 0);

    cuda_result = cuMemExportToShareableHandle((void*)&cross_memory_handles_[0].shareable_handle_, cross_memory_handles_[0].cuda_handle_, ipc_handle_type_flag_, 0);

    CUdeviceptr va_position = d_ptr; //NOTE: When having other pointers, this will adding the offsets in order to properly account for fitting into the contiguous VA range.
    cross_memory_handles_[0].vulkan_ptr = (void*)va_position;

    cuda_result = cuMemMap(va_position, cross_memory_handles_[0].size_, 0, cross_memory_handles_[0].cuda_handle_, 0);

    cuda_result = cuMemRelease(cross_memory_handles_[0].cuda_handle_);

    MemoryAccessDescriptor();

    cuMemSetAccess(d_ptr, total_alloc_size_, &access_descriptor_, 1); //Adds read-write access to the whole VA range.

    return cuda_status;
}

CUresult CudaInterop::Clean() {
    CUresult cuda_result;
    for (const auto& mem_handle : cross_memory_handles_) { //Ensures that all allocations are mapped before attempting unmap memory
        if (!mem_handle.vulkan_ptr) {
            return;
        }
    }

    IPCCloseShareableHandle(cross_memory_handles_[0].shareable_handle_);

    cuda_result = cuMemAddressFree((CUdeviceptr) cross_memory_handles_[0].vulkan_ptr, total_alloc_size_);
}

int CudaInterop::IPCCloseShareableHandle(ShareableHandle sh_handle) {
    return CloseHandle(sh_handle);
}