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

	void operator=(const InteropMemoryHandler&) = delete;

	static InteropMemoryHandler& Get() {
		static InteropMemoryHandler instance_;
		return instance_;
	}

	static CUresult CreateNewAllocation();

	static CUresult MapExistingPointer(void* ptr, const size_t size, const size_t type_size);

	static CUresult Clean();

	static inline vector<CrossMemoryHandle>& CrossMemoryHandles() {
		return Get().cross_memory_handles_;
	}

	static inline void AddMemoryHandle(const size_t size, const size_t type_size, const bool queue_submit = true) {  //NOTE: ALLOCATE STRUCT WITH SIZE
		auto memory_handle = CrossMemoryHandle(size, type_size);
		memory_handle.granularity_size = RoundWarpGranularity(size, Get().granularity);

		if (queue_submit) {
			Get().allocation_queue_.push_back(memory_handle);
		}
		else {
			Get().cross_memory_handles_.push_back(memory_handle);
		}
	}

	static void DebugGPU(CUdeviceptr& va_ptr, CrossMemoryHandle& memory_handle, const unsigned int index);

private:
	InteropMemoryHandler();
	CUresult GetAllocationGranularity(const CUmemAllocationGranularity_flags flags = CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);

	void GetDefaultSecurityDescriptor(CUmemAllocationProp* prop);

	static size_t CalculateTotalMemorySize(vector<CrossMemoryHandle>& memory_handles, const size_t granularity);

	static inline size_t RoundWarpGranularity(const size_t size, const size_t granularity) {
		return ((size + granularity - 1) / granularity) * granularity;
	}

	static CUresult AllocateMMAP(CUdeviceptr& va_position, CrossMemoryHandle& memory_handle);

	int cuda_device_ = -1, device_count_ = 0;

	size_t granularity = 0;
	size_t total_granularity_size_ = 0;

	OperatingSystem os_;

	CUmemAllocationHandleType ipc_handle_type_flag_ = {};
	CUmemAllocationProp current_alloc_prop_ = {};
	CUmemAccessDesc access_descriptor_ = {};

	vector<CUdeviceptr> va_ptrs_;
	vector<CrossMemoryHandle> cross_memory_handles_;
	vector<CrossMemoryHandle> allocation_queue_;
};