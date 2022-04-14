#include "cuda_runtime.h"
#include "handler_methods.hpp"

#include <stdexcept>

using std::string;

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

template <typename T>
cudaError_t CudaMemoryAllocator(T* ptrs[], size_t size_alloc, size_t ptr_num) {
    cudaError_t output_status;

    for (int i = 0; i < ptr_num; i++) {
        T* currentPtr = reinterpret_cast<T*>(ptr[i]);
        output_status = ((void**)&currentPtr, size_alloc * sizeof(T));
        CudaExceptionHandler(output_status, "cudaMalloc failed!");
    }

    return output_status;
}
