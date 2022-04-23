#pragma once

#include "cuda_runtime.h"

#include <vector>
#include <string>
#include <functional>

using std::string;
using std::reference_wrapper;
using std::vector;
using std::function;

/// <summary> 
/// Throws an invalid argument with the supplied error message when CUDA status is invalid.
/// </summary>
void CudaExceptionHandler(cudaError_t cuda_status, string error_message);

/// <summary> 
/// Frees an array of void pointers.
/// </summary>
void CudaMemoryFreer(void* ptrs[]);

/// <summary> 
/// Polymorphism, frees a referenced std::vector of referenced wrapped int pointers.
/// </summary>
void CudaMemoryFreer(vector<reference_wrapper<float*>>& ptrs);

/// <summary> 
/// Allocates memory to a referenced std::vector of referenced wrapped in pointers.
/// <para> Note: element_alloc does not have to be supplied if size_alloc contains the combined size of the elements. </para>
/// </summary>
void CudaMemoryAllocator(vector<reference_wrapper<float*>>& ptrs, size_t size_alloc, size_t element_alloc = 1);

/// <summary> 
/// Copies either CUDA memory into system RAM, or vice versa. Also includes error checking.
/// </summary>
cudaError_t CopyFunction(string err_msg, void* tgt, const void* src, cudaMemcpyKind mem_copy_type,
    cudaError_t error, size_t size_alloc, size_t element_alloc = 1);

/// <summary> 
/// Wraps any cudaError_t returning call to handler internal error checking
/// </summary>
cudaError_t WrapperFunction(function<cudaError_t()> func, string operation_name, string method_name, cudaError_t error);