#include "handler_methods.hpp"

#include <stdexcept>
#include <iostream>

using std::string;
using std::vector;
using std::reference_wrapper;

void CudaExceptionHandler(cudaError_t cuda_status, string error_message) {
    if (cuda_status != cudaSuccess) {
        throw std::invalid_argument(error_message);
    }
}

void CudaMemoryFreer(void* ptrs[]) {
    try {
        for (size_t i = 0; i < sizeof(ptrs); i++) {
            cudaFree(ptrs[i]);
        }
    }
    catch (std::exception e) {
        printf(e.what());
    }
}

void CudaMemoryFreer(vector<reference_wrapper<int*>>& ptrs) {
    try {
        for (size_t i = 0; i < ptrs.size(); i++) {
            cudaFree(ptrs.at(i).get());
        }
    }
    catch (std::exception e) {
        printf(e.what());
    }
}

void CudaMemoryAllocator(vector<reference_wrapper<int*>>& ptrs, size_t size_alloc) {
    for (size_t i = 0; i < ptrs.size(); i++) {
        if (ptrs.at(i).get() == nullptr) {
            cudaError_t output_status = cudaMalloc((void**)&ptrs.at(i).get(), size_alloc * sizeof(int));
            CudaExceptionHandler(output_status, "cudaMalloc failed!");
        }
    }
}