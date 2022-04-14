#pragma once

#include "cuda_runtime.h"

#include <vector>
#include <string>
#include <memory>
#include <functional>

using std::string;
using std::vector;
using std::reference_wrapper;

//Exceptions
void CudaExceptionHandler(cudaError_t cuda_status, string error_message);

//Memory Management Methods
void CudaMemoryFreer(void* ptrs[]);

void CudaMemoryFreer(vector<reference_wrapper<int*>>& ptrs);

void CudaMemoryAllocator(vector<reference_wrapper<int*>>& ptrs, size_t size_alloc);

