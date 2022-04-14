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

void CudaMemoryAllocator(vector<reference_wrapper<int*>>& ptrs, size_t size_alloc, size_t element_alloc) { //0 if used combined alloc
    for (size_t i = 0; i < ptrs.size(); i++) {
        if (ptrs.at(i).get() == nullptr) {
            cudaError_t output_status = cudaMalloc((void**)&ptrs.at(i).get(), size_alloc * element_alloc);
            CudaExceptionHandler(output_status, "cudaMalloc failed!");
        }
    }
}

cudaError_t CopyFunction(string err_msg, void* tgt, const void* src, cudaMemcpyKind mem_copy_type, 
    cudaError_t error, size_t size_alloc, size_t element_alloc) {
    if (error == cudaSuccess) {
        error = cudaMemcpy(tgt, src, size_alloc * element_alloc, mem_copy_type);
        if (error != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
        }
    }
    return error;
}