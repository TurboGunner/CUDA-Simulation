#include "handler_methods.hpp"

#include <stdexcept>
#include <iostream>
#include <math.h>

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

void CudaMemoryFreer(vector<reference_wrapper<float*>>& ptrs) {
    try {
        for (size_t i = 0; i < ptrs.size(); i++) {
            cudaFree(ptrs.at(i).get());
        }
    }
    catch (std::exception e) {
        printf(e.what());
    }
}

void CudaMemoryFreer(vector<reference_wrapper<float3*>>& ptrs) {
    try {
        for (size_t i = 0; i < ptrs.size(); i++) {
            cudaFree(ptrs.at(i).get());
        }
    }
    catch (std::exception e) {
        printf(e.what());
    }
}

void CudaMemoryAllocator(vector<reference_wrapper<float*>>& ptrs, size_t size_alloc, size_t element_alloc) { //0 if used combined alloc
    for (size_t i = 0; i < ptrs.size(); i++) {
        if (ptrs.at(i).get() == nullptr) {
            cudaError_t output_status = cudaMalloc((void**)&ptrs.at(i).get(), size_alloc * element_alloc);
            std::cout << "Allocated " << size_alloc * element_alloc << " bytes!" << std::endl;
            CudaExceptionHandler(output_status, "cudaMalloc failed!");
        }
    }
}

void CudaMemoryAllocator(vector<reference_wrapper<float3*>>& ptrs, size_t size_alloc, size_t element_alloc) {
    for (size_t i = 0; i < ptrs.size(); i++) {
        if (ptrs.at(i).get() == nullptr) {
            cudaError_t output_status = cudaMalloc((void**)&ptrs.at(i).get(), size_alloc * element_alloc);
            std::cout << "Allocated " << size_alloc * element_alloc << " bytes!" << std::endl;
            CudaExceptionHandler(output_status, "cudaMalloc failed!");
        }
    }
}

void MemoryFreer(void* ptrs[], size_t element_alloc) {
    std::cout << sizeof(ptrs) << std::endl;
    for (size_t i = 0; i < sizeof(ptrs) * element_alloc; i++) {
        free(ptrs[i]);
    }
}

void MemoryFreer(vector<reference_wrapper<float*>>& ptrs) {
    for (size_t i = 0; i < ptrs.size(); i++) {
        free(ptrs.at(i).get());
    }
}

void MemoryFreer(vector<reference_wrapper<float3*>>& ptrs) {
    for (size_t i = 0; i < ptrs.size(); i++) {
        free(ptrs.at(i).get());
    }
}

cudaError_t CopyFunction(string err_msg, void* tgt, const void* src, cudaMemcpyKind mem_copy_type,
    cudaError_t error, size_t size_alloc, size_t element_alloc) {

    if (error == cudaSuccess) {
        error = cudaMemcpy(tgt, src, size_alloc * element_alloc, mem_copy_type);
        if (error != cudaSuccess) {
            std::cout << err_msg << " due to error code " << error << "\n" << std::endl;
            std::cout << "Error Stacktrace: " << cudaGetErrorString(error) << "\n" << std::endl;
        }
    }
    return error;
}

cudaError_t WrapperFunction(function<cudaError_t()> func, string operation_name, string method_name, cudaError_t error, string optional_args) {
    cudaError_t cuda_status = error;
    if (cuda_status != cudaSuccess) {
        return cuda_status;
    }
    cuda_status = func();
    if (cuda_status != cudaSuccess) {
        std::cout << operation_name << " returned error code " << cuda_status << " after launching " << method_name << "\n" << std::endl;
        std::cout << "Error Stacktrace: " << cudaGetErrorString(cuda_status) << "\n" << std::endl;
        if (optional_args.size() > 0) {
            std::cout << "Additional Stacktrace: " << optional_args << std::endl;
        }
    }
    return cuda_status;
}

void ThreadAllocator(dim3& blocks, dim3& threads, const unsigned int& length, const unsigned int& threads_per_block) {
    unsigned int threads_per_dim = (int)sqrt(threads_per_block);
    unsigned int block_count = ((length + threads_per_dim) - 1) / (threads_per_dim);

    threads = dim3(threads_per_dim, threads_per_dim);
    blocks = dim3(block_count, block_count);

    std::cout << "Allocated " << threads.x * threads.y * block_count * block_count << " threads!" << std::endl;
}