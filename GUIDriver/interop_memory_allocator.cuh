#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "cross_memory_handle.cuh"

#include "../CUDATest/handler_methods.hpp"

//Logging
#include "../CUDATest/handler_classes.hpp"

#ifdef _WIN64
#include <VersionHelpers.h>
#include <winternl.h>
#include <sddl.h>
#endif

#include <vector>

using std::vector;

class InteropMemoryHandler {
public:
	InteropMemoryHandler(const InteropMemoryHandler&) = delete;

	static InteropMemoryHandler& Get();

	CUresult CreateNewAllocation();

	CUresult MapExistingPointer(void* ptr, const size_t size, const size_t type_size);

	CUresult Clean();

	vector<CrossMemoryHandle> cross_memory_handles_;

private:
	InteropMemoryHandler();

	CUresult GetAllocationGranularity(const CUmemAllocationGranularity_flags flags = CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);

	void GetDefaultSecurityDescriptor(CUmemAllocationProp* prop);

	void AutoExpand() {	}

	size_t CalculateTotalMemorySize(const vector<CrossMemoryHandle>& memory_handles, const size_t granularity);

	inline size_t RoundWarpGranularity(const size_t& size, const size_t& granularity) {
		return ((size + granularity - 1) / granularity) * granularity;
	}

	inline void AddMemoryHandle(const size_t size, const size_t type_size, const bool queue_submit = true) {  //NOTE: ALLOCATE STRUCT WITH SIZE
		auto memory_handle = CrossMemoryHandle(size, type_size);

		cross_memory_handles_.push_back(memory_handle);
		if (queue_submit) {
			allocation_queue_.push_back(memory_handle);
		}
	}

	int cuda_device_ = -1;

	size_t granularity = 0;
	size_t total_granularity_size_ = 0;

	OperatingSystem os_;

	CUmemAllocationHandleType ipc_handle_type_flag_ = {};
	CUmemAllocationProp current_alloc_prop_ = {};
	CUmemAccessDesc access_descriptor_ = {};

	vector<CUdeviceptr> va_ptrs_;
	vector<CrossMemoryHandle> allocation_queue_;
};