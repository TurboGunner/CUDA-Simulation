#include "interop_memory_allocator.cuh"

InteropMemoryHandler::InteropMemoryHandler() {
	cudaError_t cuda_status = cudaSuccess;

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

	cuda_status = cudaGetDevice(&cuda_device_); //Inits device

	//Allocates CUDA driver allocation prop settings
	current_alloc_prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	current_alloc_prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	current_alloc_prop_.location.id = cuda_device_;
	current_alloc_prop_.win32HandleMetaData = nullptr;
	current_alloc_prop_.requestedHandleTypes = ipc_handle_type_flag_;

	//Allocates CUDA driver access descriptor settings
	access_descriptor_ = {};
	access_descriptor_.location.id = cuda_device_;
	access_descriptor_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	access_descriptor_.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

	if (os_ != LINUX) {
		GetDefaultSecurityDescriptor(&current_alloc_prop_);
	}

	GetAllocationGranularity();
}

InteropMemoryHandler& InteropMemoryHandler::Get() {
	static InteropMemoryHandler instance_;
	return instance_;
}

CUresult InteropMemoryHandler::GetAllocationGranularity(const CUmemAllocationGranularity_flags flags) {
	CUresult cuda_result = cuMemGetAllocationGranularity(&granularity, &current_alloc_prop_, flags);
	CudaDriverLog(cuda_result, "Allocation Granularity");

	return cuda_result;
}

void InteropMemoryHandler::GetDefaultSecurityDescriptor(CUmemAllocationProp* prop) {
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

size_t InteropMemoryHandler::CalculateTotalMemorySize(const vector<CrossMemoryHandle>& memory_handles, const size_t granularity) {
	size_t total_granularity_size = 0;

	for (auto& mem_handle : cross_memory_handles_) {
		size_t current_granularity_size = RoundWarpGranularity(mem_handle.TotalAllocationSize(), granularity);
		total_granularity_size += current_granularity_size;
		mem_handle.granularity_size = current_granularity_size;
	}

	return total_granularity_size;
}

CUresult InteropMemoryHandler::CreateNewAllocation() {
	if (allocation_queue_.size() == 0) {
		ProgramLog::OutputLine("Warning: There was no elements in the allocation queue!");
		return CUDA_SUCCESS;
	}
	//Initialize nullptr equivalent
	size_t local_total_granularity_size = CalculateTotalMemorySize(allocation_queue_, granularity);
	va_ptrs_.push_back(0U);

	//Takes the latest VA CUDA Device pointer and then reserves the total granularity size
	CUresult cuda_result = cuMemAddressReserve(&va_ptrs_[va_ptrs_.size() - 1], total_granularity_size_, granularity, 0, 0);
	CudaDriverLog(cuda_result, "MemAddressReserve");

	CUdeviceptr total = 0U;

	//Traverses through all memory handles
	for (auto& memory_handle : allocation_queue_) {
		//Creates handle, and then exports it
		cuda_result = cuMemCreate(&memory_handle.cuda_handle, memory_handle.granularity_size, &current_alloc_prop_, 0);
		CudaDriverLog(cuda_result, "MemCreate");
		cuda_result = cuMemExportToShareableHandle(&memory_handle.shareable_handle, memory_handle.cuda_handle, ipc_handle_type_flag_, 0);
		CudaDriverLog(cuda_result, "ExportToShareableHandle");

		//Adds stride to memory address
		total += memory_handle.granularity_size;

		//Sets pointer to the handle
		memory_handle.cuda_device_ptr = (void*) memory_handle.cuda_handle;

		//Maps allocation to VA handle
		cuda_result = cuMemMap(total, memory_handle.granularity_size, 0, memory_handle.cuda_handle, 0);
		CudaDriverLog(cuda_result, "MapMemory");

		//Releases memory
		cuda_result = cuMemRelease(memory_handle.cuda_handle);
		CudaDriverLog(cuda_result, "ReleaseMemory");
	}
	//Sets permission for whole VA range
	cuda_result = cuMemSetAccess(va_ptrs_[va_ptrs_.size() - 1], total_granularity_size_, &access_descriptor_, 1);
	CudaDriverLog(cuda_result, "SetMemoryAccess");
	allocation_queue_.clear();

	return cuda_result;
}

CUresult InteropMemoryHandler::MapExistingPointer(void* ptr, const size_t size, const size_t type_size) {
	if (!ptr) {
		ProgramLog::OutputLine("Warning: This pointer is null!");
	}

	va_ptrs_.push_back((uintptr_t) ptr);

	CUresult cuda_result = cuMemAddressReserve(&va_ptrs_[va_ptrs_.size() - 1], total_granularity_size_, granularity, 0, 0);
	CudaDriverLog(cuda_result, "MemAddressReserve");

	AddMemoryHandle(size, type_size, false);
	CrossMemoryHandle& current = cross_memory_handles_[cross_memory_handles_.size() - 1];

	cuda_result = cuMemCreate(&current.cuda_handle, current.granularity_size, &current_alloc_prop_, 0);
	CudaDriverLog(cuda_result, "MemCreate");
	cuda_result = cuMemExportToShareableHandle(&current.shareable_handle, current.cuda_handle, ipc_handle_type_flag_, 0);
	CudaDriverLog(cuda_result, "ExportToShareableHandle");

	current.cuda_device_ptr = (void*) current.cuda_handle;

	CUdeviceptr va_position = (uintptr_t) va_ptrs_[va_ptrs_.size() - 1];

	cuda_result = cuMemMap(va_position, current.granularity_size, 0, current.cuda_handle, 0);
	CudaDriverLog(cuda_result, "MapMemory");

	cuda_result = cuMemRelease(current.cuda_handle);
	CudaDriverLog(cuda_result, "ReleaseMemory");

	cuda_result = cuMemSetAccess(va_ptrs_[va_ptrs_.size() - 1], total_granularity_size_, &access_descriptor_, 1);
	CudaDriverLog(cuda_result, "SetMemoryAccess");

	return cuda_result;
}

CUresult InteropMemoryHandler::Clean() {
	CUresult cuda_result;
	for (const auto& mem_handle : cross_memory_handles_) { //Ensures that all allocations are mapped before attempting to unmap memory
		if (!mem_handle.vulkan_ptr) {
			CudaDriverLog(cuda_result, "Clean");
			return cuda_result;
		}
	}
	for (const auto& mem_handle : cross_memory_handles_) {
		CloseHandle(mem_handle.shareable_handle);

		cuda_result = cuMemAddressFree((CUdeviceptr) mem_handle.cuda_device_ptr, total_granularity_size_);
		CudaDriverLog(cuda_result, "VulkanPtrCUDAFree");
	}
	return cuda_result;
}