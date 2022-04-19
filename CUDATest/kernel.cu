
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "handler_methods.hpp"
#include "vector_field.hpp"
#include "fluid_sim.hpp"

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

    // Add vectors in parallel
    cudaError_t cudaStatus = AddWithCuda(c, a, b, arraySize);
    CudaExceptionHandler(cudaStatus, "addWithCuda failed!");

    printf("{1,2,3,4,5} * {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    cudaStatus = cudaDeviceReset();
    CudaExceptionHandler(cudaStatus, "cudaDeviceReset failed!");

    VectorField field(3, 3);
    FluidSim simulation(.1f, 1.0f, 1, 3, 3, 32);

    simulation.AddVelocity(IndexPair(0, 0), 10, 10);
    simulation.AddVelocity(IndexPair(1, 0), 2, 2);
    simulation.AddVelocity(IndexPair(1, 1), 20, 20);
    simulation.AddDensity(IndexPair(1, 1), 1.0f, 1.0f);
    simulation.AddDensity(IndexPair(2, 2), 10.0f, 1.0f);

    simulation.Diffuse(1, 1.0f, 1.0f);
    simulation.Project();
    simulation.Advect(0, 1.0f);
    std::cout << simulation.density_.ToString() << std::endl;
    std::cout << simulation.velocity_.ToString() << std::endl;

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
        return cudaStatus;
    }

    CudaMemoryAllocator(bidoof, (size_t) size, sizeof(int));
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = CopyFunction("cudaMemcpy failed!", dev_a, a, cudaMemcpyHostToDevice, cudaStatus, (size_t)size, sizeof(int));
    cudaStatus = CopyFunction("cudaMemcpy failed!", dev_b, b, cudaMemcpyHostToDevice, cudaStatus, (size_t)size, sizeof(int));

    // Launch a kernel on the GPU with one thread for each element.
    AddKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        CudaMemoryFreer(bidoof);
        return cudaStatus;
    }
    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        CudaMemoryFreer(bidoof);
        return cudaStatus;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = CopyFunction("cudaMemcpy failed!", c, dev_c, cudaMemcpyDeviceToHost, cudaStatus, (size_t)size, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        CudaMemoryFreer(bidoof);
    }
    return cudaStatus;
}