#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "vulkan/vulkan.h"

#include "handler_classes.hpp"

#include <vector>
#include <string>
#include <functional>
#include <fstream>

using std::string;
using std::reference_wrapper;
using std::vector;
using std::function;

/// <summary> Throws an invalid argument with the supplied error message when CUDA status is invalid. </summary>
void CudaExceptionHandler(const cudaError_t& cuda_status, string error_message);

void VulkanExceptionHandler(const VkResult& vulkan_status, const string& error_message);

__host__ __device__ void CudaExceptionHandlerCross(cudaError_t cuda_status, const char* error_message);

void ErrorLog(const cudaError_t& cuda_status, const string& operation_name, const string& method_name, const string& optional_args = "");

void ErrorLog(const VkResult& vk_status, const string& operation_name, const string& method_name, const string& optional_args = "");

/// <summary> Allocates memory to a referenced std::vector of referenced wrapped generic pointers.
/// <para> Note: element_alloc does not have to be supplied if size_alloc contains the combined size of the elements. </para> </summary>
template <typename T>
void CudaMemoryAllocator(vector<reference_wrapper<T*>>& ptrs, size_t size_alloc, size_t element_alloc = 1);

/// <summary> Copies either CUDA memory into system RAM, or vice versa. Also includes error checking. </summary>
cudaError_t CopyFunction(string err_msg, void* tgt, const void* src, cudaMemcpyKind mem_copy_type, cudaError_t error, size_t size_alloc, size_t element_alloc = 1);

/// <summary> Wraps any cudaError_t returning call to handler internal error checking. </summary>
cudaError_t WrapperFunction(function<cudaError_t()> func, string operation_name, string method_name, cudaError_t error, string optional_args = "");

/// <summary> Allocates threads based on given vector field lengths and threads allocated per dimension. Order: blocks, threads. </summary>
void ThreadAllocator(dim3& blocks, dim3& threads, const unsigned int& length, const unsigned int& threads_per_block = 16);

cudaError_t PostExecutionChecks(cudaError_t status, string method_name, bool sync_wait = false);

/// <summary> Allocates threads based on given vector field lengths and threads allocated per dimension. Order: blocks, threads. 
/// <para> Delegated specificially for the BoundaryConditions check. </para> </summary>
void ThreadAllocator2D(dim3& blocks, dim3& threads, const unsigned int& length, const unsigned int& threads_per_block = 16);

__host__ void CudaDriverLog(CUresult& cuda_result, const string& label = "");