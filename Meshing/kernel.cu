
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "matrix.cuh"

#include "../CUDATest/handler_methods.hpp"

#include <stdio.h>

#include <iostream>
#include <functional>

using std::function;

int main() {
    cudaError_t cuda_status = cudaSuccess;

    function<cudaError_t()> set_device_func = []() { return cudaSetDevice(0); };
    cuda_status = WrapperFunction(set_device_func, "cudaSetDevice failed!", "main",
        cuda_status, "Do you have a CUDA-capable GPU installed?");

    Matrix* matrix = Matrix::Create(3, 3, true);

    for (size_t i = 0; i < matrix->rows; i++) {
        for (size_t j = 0; j < matrix->columns; j++) {
            matrix->Set(2.0f + ((i + 5) + (j * (j + 1))), i, j);
        }
    }

    matrix->Set(10.0f, 0, 0);

    for (size_t i = 0; i < matrix->rows; i++) {
        std::cout << std::endl;
        for (size_t j = 0; j < matrix->columns; j++) {
            std::cout << matrix->Get(i, j) << " ";
        }
    }

    Matrix transpose = matrix->Transpose();

    std::cout << std::endl << std::endl;

    for (size_t i = 0; i < transpose.rows; i++) {
        std::cout << std::endl;
        for (size_t j = 0; j < transpose.columns; j++) {
            std::cout << transpose.Get(i, j) << " ";
        }
    }

    std::cout << std::endl << std::endl;

    Matrix adjoint(matrix->rows, matrix->columns, true);
    Matrix::Adjoint(*matrix, adjoint);

    for (size_t i = 0; i < adjoint.rows; i++) {
        std::cout << std::endl;
        for (size_t j = 0; j < adjoint.columns; j++) {
            std::cout << adjoint.Get(i, j) << " ";
        }
    }

    std::cout << std::endl << std::endl;

    Matrix inverse(matrix->rows, matrix->columns, true);
    Matrix::Inverse(*matrix, inverse);

    for (size_t i = 0; i < inverse.rows; i++) {
        std::cout << std::endl;
        for (size_t j = 0; j < inverse.columns; j++) {
            std::cout << inverse.Get(i, j) << " ";
        }
    }

    cuda_status = cudaDeviceReset();
    CudaExceptionHandler(cuda_status, "cudaDeviceReset failed!");

    return 0;
}