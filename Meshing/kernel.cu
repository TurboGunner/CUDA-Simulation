
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "matrix.cuh"

#include "../CUDATest/handler_methods.hpp"
#include "../CUDATest/handler_classes.hpp"

#include <stdio.h>

#include <iostream>
#include <functional>
#include <vector>

using std::function;
using std::vector;

Matrix* GMatrixTerm(Matrix* matrix, Matrix* matrix_T, Matrix* weights) {
    Matrix* transpose_weights = Matrix::MultiplyGPU(matrix_T, weights);
    Matrix* multiply = Matrix::MultiplyGPU(transpose_weights, weights);

   return Matrix::MultiplyGPU(multiply, matrix);
}

Matrix* Weights(Matrix* matrix) {
    vector<float> arr;
    for (size_t i = 0; i < matrix->rows; i++) {
        float* ptr = matrix->Row(i);
        arr.push_back(sqrt(pow(ptr[0], 2) + pow(ptr[1], 2) + pow(ptr[2], 2)));
        free(ptr);
    }

    Matrix* diagonal = Matrix::DiagonalMatrix(arr.data(), 4, 4);
    *diagonal = diagonal->Reciprocal();

    diagonal->DeviceTransfer(diagonal->device_alloc, diagonal);

    return diagonal;
}

int main() {
    cudaError_t cuda_status = cudaSuccess;

    function<cudaError_t()> set_device_func = []() { return cudaSetDevice(0); };
    cuda_status = WrapperFunction(set_device_func, "cudaSetDevice failed!", "main",
        cuda_status, "Do you have a CUDA-capable GPU installed?");

    Matrix* matrix = Matrix::Create(4, 3, false);

    Matrix::PopulateRandomHost(matrix, 0.0f, 1.0f);

    matrix->PrintMatrix("Matrix: ");

    Matrix* transpose = Matrix::TransposeGPU(matrix);
    transpose->PrintMatrix("Transpose: ");

    Matrix* weights = Weights(matrix);

    weights->PrintMatrix("Weights: ");

    Matrix* G_term = GMatrixTerm(matrix, transpose, weights);

    G_term->PrintMatrix("G Term: ");

    Matrix* inverse = Matrix::Create(G_term->rows, G_term->columns);
    Matrix::Inverse(*G_term, *inverse);

    inverse->DeviceTransfer(inverse->device_alloc, inverse);

    inverse->PrintMatrix("Inverse of G Term: ");

    Matrix* multiply_t = Matrix::MultiplyGPU(inverse, transpose);

    Matrix* weight_multiply = Matrix::MultiplyGPU(multiply_t, weights);

    Matrix* gradient = Matrix::MultiplyGPU(weight_multiply, weights);
    gradient->PrintMatrix("Gradient: ");

    RandomFloat random(100.0f, 500.0f, 1);

    Matrix* delta = Matrix::Create(4, 1);
    for (size_t j = 0; j < delta->rows; j++) {
        delta->Set(random.Generate(), j);
    }

    delta->PrintMatrix("Temperature Delta: ");

    delta->DeviceTransfer(delta->device_alloc, delta);
    Matrix* multiply_t4 = Matrix::MultiplyGPU(gradient, delta);

    multiply_t4->PrintMatrix("Gradient Solution: ");

    cuda_status = cudaDeviceReset();
    CudaExceptionHandler(cuda_status, "cudaDeviceReset failed!");

    return 0;
}