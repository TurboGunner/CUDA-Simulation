#include "interop_memory_allocator.cuh"
#include "../Meshing/vector_cross.cuh"

InteropMemoryHandler::InteropMemoryHandler() {
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

	
	cudaError_t cuda_status = cudaGetDeviceCount(&device_count_); //Gets current CUDA device

	if (device_count_ > 1) {
		ProgramLog::OutputLine("Warning: There are multiple CUDA devices!");
	}

	cuda_status = cudaGetDevice(&cuda_device_);

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

CUresult InteropMemoryHandler::GetAllocationGranularity(const CUmemAllocationGranularity_flags flags) {
	CUresult cuda_result = cuMemGetAllocationGranularity(&granularity, &current_alloc_prop_, flags);
	CudaDriverLog(cuda_result, "Allocation Granularity");

	ProgramLog::OutputLine("Allocation Granularity: " + std::to_string(granularity));

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

size_t InteropMemoryHandler::CalculateTotalMemorySize(vector<CrossMemoryHandle>& memory_handles, const size_t granularity) {
	size_t total_granularity_size = 0;

	for (auto& mem_handle : memory_handles) {
		size_t current_granularity_size = RoundWarpGranularity(mem_handle.TotalAllocationSize(), granularity);
		total_granularity_size += current_granularity_size;
		mem_handle.granularity_size = current_granularity_size;
	}

	return total_granularity_size;
}

CUresult InteropMemoryHandler::CreateNewAllocation() {
	if (Get().allocation_queue_.size() == 0) {
		ProgramLog::OutputLine("Warning: There was no elements in the allocation queue!");
		return CUDA_SUCCESS;
	}
	//Initialize nullptr equivalent
	Get().total_granularity_size_ = CalculateTotalMemorySize(Get().allocation_queue_, Get().granularity);
	Get().va_ptrs_.push_back(0U);

	//Takes the latest VA CUDA Device pointer and then reserves the total granularity size
	CUresult cuda_result = cuMemAddressReserve(&Get().va_ptrs_[Get().va_ptrs_.size() - 1], Get().total_granularity_size_, Get().granularity, 0, 0);
	CudaDriverLog(cuda_result, "MemAddressReserve");

	CUdeviceptr total = Get().va_ptrs_[Get().va_ptrs_.size() - 1];

	//Traverses through all memory handles
	for (auto& memory_handle : Get().allocation_queue_) {
		AllocateMMAP(total, memory_handle);
		total += memory_handle.granularity_size; //Adds stride to memory address
	}

	ProgramLog::OutputLine("Device Pointer on First CrossMemoryHandle: " + std::to_string((uintptr_t) Get().allocation_queue_[0].cuda_device_ptr));

	//Sets permission for whole VA range
	cuda_result = cuMemSetAccess(Get().va_ptrs_[Get().va_ptrs_.size() - 1], Get().total_granularity_size_, &Get().access_descriptor_, 1);
	CudaDriverLog(cuda_result, "SetMemoryAccess");
	Get().cross_memory_handles_.insert(Get().cross_memory_handles_.begin(), Get().allocation_queue_.begin(), Get().allocation_queue_.end());
	Get().allocation_queue_.clear();

	return cuda_result;
}

CUresult InteropMemoryHandler::MapExistingPointer(void* ptr, const size_t size, const size_t type_size) {
	if (!ptr) {
		ProgramLog::OutputLine("Warning: This pointer is null!");
	}

	Get().AddMemoryHandle(size, type_size, false);
	CrossMemoryHandle& current = Get().cross_memory_handles_[Get().cross_memory_handles_.size() - 1];

	Get().total_granularity_size_ = CalculateTotalMemorySize(Get().cross_memory_handles_, Get().granularity);

	Get().va_ptrs_.push_back(0U);

	size_t granularity_offset = ((uintptr_t) ptr % Get().granularity) * Get().granularity;

	s_stream << "Device Pointer Address: " << (uintptr_t) ptr;
	ProgramLog::OutputLine(s_stream);
	ProgramLog::OutputLine("VA Mapping Granularity Offset: " + std::to_string(granularity_offset));
	size_t alignment = Get().granularity;

	CUresult cuda_result = cuMemAddressReserve(&Get().va_ptrs_[Get().va_ptrs_.size() - 1], Get().total_granularity_size_, alignment, granularity_offset, 0);
	CudaDriverLog(cuda_result, "MemAddressReserve");

	CUdeviceptr va_position = (uintptr_t) Get().va_ptrs_[Get().va_ptrs_.size() - 1];
	cuda_result = AllocateMMAP(va_position, current);

	s_stream << "Pointer Test (VA): " << Get().va_ptrs_[Get().va_ptrs_.size() - 1];
	ProgramLog::OutputLine(s_stream);

	cuda_result = cuMemSetAccess(Get().va_ptrs_[Get().va_ptrs_.size() - 1], Get().total_granularity_size_, &Get().access_descriptor_, 1);
	CudaDriverLog(cuda_result, "SetMemoryAccess");

	DebugGPU(Get().va_ptrs_[Get().va_ptrs_.size() - 1], current, 163);

	return cuda_result;
}

CUresult InteropMemoryHandler::AllocateMMAP(CUdeviceptr& va_position, CrossMemoryHandle& memory_handle) {
	CUresult cuda_result;
	cuda_result = cuMemCreate(&memory_handle.cuda_handle, memory_handle.granularity_size, &Get().current_alloc_prop_, 0);
	CudaDriverLog(cuda_result, "MemCreate");
	cuda_result = cuMemExportToShareableHandle(&memory_handle.shareable_handle, memory_handle.cuda_handle, Get().ipc_handle_type_flag_, 0);
	CudaDriverLog(cuda_result, "ExportToShareableHandle");

	//Sets pointer to the handle
	memory_handle.cuda_device_ptr = (void*) va_position;

	//Maps allocation to VA handle
	cuda_result = cuMemMap(va_position, memory_handle.granularity_size, 0, memory_handle.cuda_handle, 0);
	CudaDriverLog(cuda_result, "MapMemory");

	//Releases memory
	cuda_result = cuMemRelease(memory_handle.cuda_handle);
	CudaDriverLog(cuda_result, "ReleaseMemory");

	return cuda_result;
}

CUresult InteropMemoryHandler::Clean() {
	CUresult cuda_result;
	for (const auto& mem_handle : Get().cross_memory_handles_) { //Ensures that all allocations are mapped before attempting to unmap memory
		if (!mem_handle.cuda_device_ptr) {
			CudaDriverLog(cuda_result, "Clean");
			return cuda_result;
		}
	}
	for (const auto& mem_handle : Get().cross_memory_handles_) {
		CloseHandle(mem_handle.shareable_handle);

		cuda_result = cuMemAddressFree((CUdeviceptr) mem_handle.cuda_device_ptr, Get().total_granularity_size_);
		CudaDriverLog(cuda_result, "VulkanPtrCUDAFree");
	}
	return cuda_result;
}

void InteropMemoryHandler::DebugGPU(CUdeviceptr& va_ptr, CrossMemoryHandle& memory_handle, const unsigned int index) {
	if (memory_handle.size <= index) {
		ProgramLog::OutputLine("Warning: Invalid Index! No debug will be called.");
		return;
	}

	Vector3D* host_test = new Vector3D();
	Vector3D* device_test_host = new Vector3D(1, 1, 1);

	Vector3D* vector_device_ptr = (Vector3D*)memory_handle.cuda_device_ptr;
	Vector3D* vector_va_ptr = (Vector3D*)va_ptr;

	cudaError_t cuda_status = cudaMemcpy(&(vector_device_ptr[163]), device_test_host, sizeof(Vector3D), cudaMemcpyHostToDevice);
	cuda_status = cudaMemcpy(host_test, &(vector_va_ptr[163]), sizeof(Vector3D), cudaMemcpyDeviceToHost);

	s_stream << "Pointer Test: " << host_test->dim[0] << std::endl;
	ProgramLog::OutputLine(s_stream);

	delete host_test;
	delete device_test_host;
}