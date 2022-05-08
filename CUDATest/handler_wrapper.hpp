#pragma once

#include "cuda_runtime.h"

#include "handler_methods.hpp"
#include "vector_field.hpp"

#include <vector>
#include <string>
#include <functional>

using std::string;
using std::reference_wrapper;
using std::vector;
using std::function;

class CudaMethodHandler {
	public:
		CudaMethodHandler() = default;
		CudaMethodHandler(const unsigned int& alloc_size, string method_name);

		~CudaMethodHandler();

		void AllocateCopyPointers();

		cudaError_t CopyToMemory(cudaMemcpyKind mode);

		cudaError_t PostExecutionChecks();

		vector<reference_wrapper<float*>> float_ptrs_, float_copy_ptrs_;
		vector<reference_wrapper<float3*>> float3_ptrs_, float3_copy_ptrs_;

		string method_name_;

		unsigned int alloc_size_;
};