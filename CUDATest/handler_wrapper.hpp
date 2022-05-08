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
		/// <summary> 
		/// Default constructor, unassigned method name and allocation size
		/// </summary>
		CudaMethodHandler() = default;

		/// <summary> 
		/// Loaded constructor. Takes in allocation size (overall size of the vector field) and the name of the method being executed.
		/// </summary>
		CudaMethodHandler(const unsigned int& alloc_size, string method_name);

		/// <summary> 
		/// Destructor. Frees supplied GPU and system memory pointers separately.
		/// </summary>
		~CudaMethodHandler();

		/// <summary> 
		/// Allocates supplied copy pointers.
		/// </summary>
		void AllocateCopyPointers();

		/// <summary> 
		/// Copies all supplied system and GPU copy pointers based on the supplied cudaMemcpyKind enum.
		/// <para> Recommended for host to device transactions only due to lack of ability to specify which should be copied. </para>
		/// </summary>
		cudaError_t CopyToMemory(cudaMemcpyKind mode, cudaError_t status);

		/// <summary> 
		/// Runs last error check and CUDA synchronization. Should be made after the kernel call of the global method.
		/// </summary>
		cudaError_t PostExecutionChecks(cudaError_t status);

		vector<reference_wrapper<float*>> float_ptrs_, float_copy_ptrs_;
		vector<reference_wrapper<float3*>> float3_ptrs_, float3_copy_ptrs_;

		string method_name_;

		unsigned int alloc_size_;
};