#include "handler_methods.hpp"

#include <stdexcept>
#include <iostream>
#include <math.h>

void CudaExceptionHandler(const cudaError_t& cuda_status, string error_message) {
    if (cuda_status != cudaSuccess) {
        s_stream << "Error Stacktrace: " << cudaGetErrorString(cuda_status) << "\n" << std::endl;
        ProgramLog::OutputLine(s_stream);

        throw std::invalid_argument(error_message);
    }
}

void VulkanExceptionHandler(const VkResult& vulkan_status, const string& error_message) {
    if (vulkan_status != VK_SUCCESS) {
        s_stream << "Error: " << vulkan_status << "\n" << std::endl;
        ProgramLog::OutputLine(s_stream);

        throw std::invalid_argument(error_message);
    }
}

__host__ __device__ void CudaExceptionHandlerCross(cudaError_t cuda_status, const char* error_message) {
    if (cuda_status != cudaSuccess) {
        printf("%s\n", error_message);
        printf("Error Stacktrace: %s\n\n", cudaGetErrorString(cuda_status));
    }
}

void ErrorLog(const cudaError_t& cuda_status, const string& operation_name, const string& method_name, const string& optional_args) {
    if (cuda_status != cudaSuccess) {
        std::cout << operation_name << " returned error code " << cuda_status << " after launching " << method_name << "\n" << std::endl;
        std::cout << "Error Stacktrace: " << cudaGetErrorString(cuda_status) << "\n" << std::endl;
        if (optional_args.size() > 0) {
            s_stream << "Additional Stacktrace: " << optional_args << std::endl;
            ProgramLog::OutputLine(s_stream);
        }
    }
}

void ErrorLog(const VkResult& vk_status, const string& operation_name, const string& method_name, const string& optional_args) {
    if (vk_status != VK_SUCCESS) {
        std::cout << operation_name << " returned error code " << vk_status << " after launching " << method_name << "\n" << std::endl;
        std::cout << "Error Stacktrace: " << vk_status << "\n" << std::endl;
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