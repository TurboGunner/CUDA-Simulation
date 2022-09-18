
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "matrix.cuh"

#include "../CUDATest/handler_methods.hpp"
#include "../CUDATest/handler_classes.hpp"

#include <stdio.h>

#include <iostream>
#include <functional>

using std::function;

Matrix* GMatrixTerm(Matrix* matrix, Matrix* matrix_T, Matrix* weights) {
    std::cout << std::endl << std::endl;

    Matrix* multiply_t = Matrix::MultiplyGPU(matrix_T, weights);
    multiply_t->PrintMatrix();
    std::cout << std::endl << std::endl;
    Matrix* multiply = Matrix::MultiplyGPU(multiply_t, weights);
    multiply->PrintMatrix();
    std::cout << std::endl << std::endl;

   return Matrix::MultiplyGPU(multiply, matrix);
}

Matrix* Weights(Matrix* matrix) {
    vector<float> arr;
    for (size_t i = 0; i < matrix->rows; i++) {
        float* ptr = matrix->Row(i);
        arr.push_back(sqrt(pow(ptr[0], 2) + pow(ptr[1], 2) + pow(ptr[2], 2)));
        free(ptr);
    }

    std::cout << std::endl << std::endl;

    Matrix* diagonal = Matrix::DiagonalMatrix(arr.data(), 4, 4);
    *diagonal = diagonal->Reciprocal();

    std::cout << std::endl << std::endl;
    diagonal->PrintMatrix();
    std::cout << std::endl << std::endl;
    diagonal->DeviceTransfer(diagonal->device_alloc, diagonal);

    return diagonal;
}

cudaError_t PopulateMatrix(Matrix* matrix) {
    RandomFloat random(0, 1.0f, 3);

    for (size_t i = 0; i < matrix->rows; i++) {
        for (size_t j = 0; j < matrix->columns; j++) {
            matrix->Set(random.Generate(), j, i);
        }
    }

    return matrix->DeviceTransfer(matrix->device_alloc, matrix);
}

int main() {
    cudaError_t cuda_status = cudaSuccess;

    function<cudaError_t()> set_device_func = []() { return cudaSetDevice(0); };
    cuda_status = WrapperFunction(set_device_func, "cudaSetDevice failed!", "main",
        cuda_status, "Do you have a CUDA-capable GPU installed?");

    Matrix* matrix = Matrix::Create(4, 3, false);

    PopulateMatrix(matrix);

    matrix->PrintMatrix();

    std::cout << std::endl << std::endl;

    Matrix* transpose = Matrix::TransposeGPU(matrix);
    transpose->PrintMatrix();

    std::cout << std::endl << std::endl;

    Matrix* weights = Weights(matrix);

    weights->PrintMatrix();

    std::cout << std::endl << "G: " << std::endl;

    Matrix* G_term = GMatrixTerm(matrix, transpose, weights);

    G_term->PrintMatrix();

    std::cout << std::endl << std::endl;

    Matrix* inverse = Matrix::Create(G_term->rows, G_term->columns);
    Matrix::Inverse(*G_term, *inverse);

    inverse->DeviceTransfer(inverse->device_alloc, inverse);

    inverse->PrintMatrix();

    std::cout << std::endl << std::endl;

    Matrix* multiply_t = Matrix::MultiplyGPU(inverse, transpose);
    multiply_t->PrintMatrix();
    std::cout << std::endl << std::endl;
    Matrix* multiply_t2 = Matrix::MultiplyGPU(multiply_t, weights);
    Matrix* multiply_t3 = Matrix::MultiplyGPU(multiply_t2, weights);
    multiply_t3->PrintMatrix();

    RandomFloat random(100.0f, 500.0f, 1);

    std::cout << std::endl << std::endl;

    Matrix* delta = Matrix::Create(4, 1);
    for (size_t j = 0; j < delta->rows; j++) {
        delta->Set(random.Generate(), j);
    }

    delta->PrintMatrix();

    delta->DeviceTransfer(delta->device_alloc, delta);
    Matrix* multiply_t4 = Matrix::MultiplyGPU(multiply_t3, delta);

    std::cout << std::endl << std::endl;

    multiply_t4->PrintMatrix();

    cuda_status = cudaDeviceReset();
    CudaExceptionHandler(cuda_status, "cudaDeviceReset failed!");

    return 0;
}