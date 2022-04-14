#pragma once

#include <string>

void CudaExceptionHandler(cudaError_t cuda_status, std::string error_message);

void CudaMemoryFreer(void* ptrs[]);

template <typename T>
void CudaMemoryAllocator(void* ptrs[]);