
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "matrix.cuh"

#include "../CUDATest/handler_methods.hpp"
#include "../CUDATest/handler_classes.hpp"

#include <stdio.h>

#include <iostream>
#include <functional>

using std::function;

int main() {
    cudaError_t cuda_status = cudaSuccess;

    function<cudaError_t()> set_device_func = []() { return cudaSetDevice(0); };
    cuda_status = WrapperFunction(set_device_func, "cudaSetDevice failed!", "main",
        cuda_status, "Do you have a CUDA-capable GPU installed?");

    Matrix* matrix = Matrix::Create(3, 4, false);
    std::cout << sizeof(Matrix) << std::endl;

    RandomFloat random(0.0f, 10.0f, 3);

    for (size_t i = 0; i < matrix->rows; i++) {
        for (size_t j = 0; j < matrix->columns; j++) {
            matrix->Set(random.Generate(), i, j);
        }
    }

    matrix->PrintMatrix();

    cuda_status = matrix->DeviceTransfer(matrix->device_alloc, matrix);

    Matrix* transpose = Matrix::Create(4, 3, false);

    cuda_status = transpose->DeviceTransfer(transpose->device_alloc, transpose);

    Matrix::TransposeGPU(matrix, transpose);

    std::cout << std::endl << std::endl;

    transpose->PrintMatrix();

    std::cout << std::endl << std::endl;


    Matrix adjoint(matrix->rows, matrix->columns, true);
    Matrix::Adjoint(*matrix, adjoint);

    adjoint.PrintMatrix();

    std::cout << std::endl << std::endl;

    Matrix inverse(matrix->rows, matrix->columns, true);
    Matrix::Inverse(*matrix, inverse);

    inverse.PrintMatrix();

    std::cout << std::endl << std::endl;

    Matrix* multiply = Matrix::MultiplyGPU(matrix, transpose);
    multiply->PrintMatrix();

    cuda_status = cudaDeviceReset();
    CudaExceptionHandler(cuda_status, "cudaDeviceReset failed!");

    return 0;
}