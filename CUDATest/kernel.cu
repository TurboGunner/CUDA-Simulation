
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "handler_methods.hpp"

#include <iostream>

#include <stdio.h>

using std::vector;
using std::reference_wrapper;

cudaError_t AddWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void AddKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] * b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 },
        b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = AddWithCuda(c, a, b, arraySize);
    CudaExceptionHandler(cudaStatus, "addWithCuda failed!");

    printf("{1,2,3,4,5} * {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    cudaStatus = cudaDeviceReset();

    CudaExceptionHandler(cudaStatus, "cudaDeviceReset failed!");

    return 0;
}

cudaError_t AddWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int* dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
    cudaError_t cudaStatus;

    vector<reference_wrapper<int*>> bidoof;
    bidoof.insert(bidoof.end(), { dev_c, dev_a, dev_b });

    cudaStatus = cudaSetDevice(0); //Assumes no multi-GPU
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)

    CudaMemoryAllocator(bidoof, (size_t) size);

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    AddKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    Error:
        CudaMemoryFreer(bidoof);
    return cudaStatus;
}