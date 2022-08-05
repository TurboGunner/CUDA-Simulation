#include "handler_methods.hpp"

#include <stdexcept>
#include <iostream>
#include <math.h>

void CudaExceptionHandler(cudaError_t& cuda_status, string error_message) {
    if (cuda_status != cudaSuccess) {
        throw std::invalid_argument(error_message);
    }
}

void ErrorLog(cudaError_t cuda_status, string operation_name, string method_name, string optional_args) {
    if (cuda_status != cudaSuccess) {
        std::cout << operation_name << " returned error code " << cuda_status << " after launching " << method_name << "\n" << std::endl;
        std::cout << "Error Stacktrace: " << cudaGetErrorString(cuda_status) << "\n" << std::endl;
        if (optional_args.size() > 0) {
            std::cout << "Additional Stacktrace: " << optional_args << std::endl;
        }
    }
}

cudaError_t CopyFunction(string err_msg, void* tgt, const void* src, cudaMemcpyKind mem_copy_type, cudaError_t error, size_t size_alloc, size_t element_alloc) {
    if (error == cudaSuccess) {
        error = cudaMemcpy(tgt, src, size_alloc * element_alloc, mem_copy_type);
        ErrorLog(error, "Copy", "CopyFunction", err_msg);
    }
    return error;
}

cudaError_t WrapperFunction(function<cudaError_t()> func, string operation_name, string method_name, cudaError_t error, string optional_args) {
    cudaError_t cuda_status = error;
    if (cuda_status == cudaSuccess) {
        return cuda_status;
    }
    cuda_status = func();
    ErrorLog(error, operation_name, method_name, optional_args);
    return cuda_status;
}

void ThreadAllocator(dim3& blocks, dim3& threads, const unsigned int& length, const unsigned int& threads_per_block) {
    unsigned int threads_per_dim = (unsigned int) cbrt(threads_per_block);
    unsigned int block_count = ((length + threads_per_dim) - 1) / (threads_per_dim);

    threads = dim3(threads_per_dim, threads_per_dim, threads_per_dim);
    blocks = dim3(block_count - 1, block_count - 1, block_count - 1);
}

void ThreadAllocator2D(dim3& blocks, dim3& threads, const unsigned int& length, const unsigned int& threads_per_block) {
    unsigned int threads_per_dim = (unsigned int)sqrt(threads_per_block);
    unsigned int block_count = ((length + threads_per_dim) - 1) / (threads_per_dim);

    threads = dim3(threads_per_dim, threads_per_dim);
    blocks = dim3(block_count - 1, block_count - 1);
}


cudaError_t PostExecutionChecks(cudaError_t status, string method_name, bool sync_wait) {
    cudaError_t cuda_status = status;
    if (cuda_status == cudaSuccess) {
        function<cudaError_t()> error_check_func = []() { return cudaGetLastError(); };
        cuda_status = WrapperFunction(error_check_func, "cudaGetLastError (kernel launch)", method_name, cuda_status);
        if (sync_wait) {
            function<cudaError_t()> sync_func = []() { return cudaDeviceSynchronize(); };
            cuda_status = WrapperFunction(sync_func, "cudaDeviceSynchronize", method_name, cuda_status);
        }
    }
    return cuda_status;
}