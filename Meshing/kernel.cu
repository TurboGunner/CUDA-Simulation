
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "matrix.cuh"
#include "mpm.cuh"

#include "../CUDATest/handler_methods.hpp"
#include "../CUDATest/handler_classes.hpp"

#include <stdio.h>

#include <iostream>
#include <functional>
#include <vector>

using std::function;
using std::vector;

int main() {
    cudaError_t cuda_status = cudaSuccess;

    function<cudaError_t()> set_device_func = []() { return cudaSetDevice(0); };
    cuda_status = WrapperFunction(set_device_func, "cudaSetDevice failed!", "main",
        cuda_status, "Do you have a CUDA-capable GPU installed?");

    //Matrix* matrix = Matrix::Create(4, 3, false);

    //Matrix::WeightedLeastSquares(matrix);

    //size_t dim = 32;

    //Grid* grid = new Grid(Vector3D(dim, dim, dim), 4);

    //cuda_status = Grid::SimulateGPU(grid);

    //cuda_status = cudaDeviceReset();
    CudaExceptionHandler(cuda_status, "cudaDeviceReset failed!");

    return 0;
}