#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "fluid_sim_cuda.cuh"

#include "handler_methods.hpp"
#include "vector_field.hpp"
#include "fluid_sim.hpp"

#include <iostream>

#include <stdio.h>

using std::vector;
using std::reference_wrapper;

cudaError_t NavierStokesCuda(float* c, const float* a, const float* b, unsigned int size);

__global__ void VelocityKernel(float *c, const float *velocity_x, const float *velocity_y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 5; i += blockDim.x * gridDim.x)
    {
        c[i] = velocity_x[i] * velocity_y[i];
    }
}

void SimulationOperations(FluidSim& simulation) {
    simulation.AddVelocity(IndexPair(0, 0), 10, 10);
    simulation.AddVelocity(IndexPair(1, 0), 2, 2);
    simulation.AddVelocity(IndexPair(1, 1), 20, 20);

    simulation.AddDensity(IndexPair(1, 1), 1.0f, 1.0f);
    simulation.AddDensity(IndexPair(2, 2), 10.0f, 1.0f);

    simulation.Diffuse(1, 1.0f, 1.0f);
    simulation.Project();
    simulation.Advect(0, 1.0f);
    //std::cout << simulation.density_.ToString() << std::endl;
    //std::cout << simulation.velocity_.ToString() << std::endl;
}

int main()
{
    const int arraySize = 5;
    const float a[arraySize] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f },
        b[arraySize] = { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f };
    float c[arraySize] = { 0.0f };

    FluidSim simulation(.1f, 1.0f, 1, 4, 4, 32);
    SimulationOperations(simulation);

    // Add vectors in parallel
    cudaError_t cudaStatus;//= NavierStokesCuda(c, a, b, arraySize);

    float a_fac = simulation.dt_ * simulation.diffusion_ * (simulation.size_x_ - 2) * (simulation.size_x_ - 2);
    float c_fac = 1.0f + (4.0f * a_fac);

    float* results = LinearSolverCuda(0, simulation.density_, simulation.density_prev_, a_fac, c_fac);

    //CudaExceptionHandler(cudaStatus, "addWithCuda failed!");
    for (const float& result : c) {
        //std::cout << result << std::endl;
    }

    for (int i = 0; i < sizeof(results) * 2; i++) {
        std::cout << results[i] << std::endl;
    }

    cudaStatus = cudaDeviceReset();
    CudaExceptionHandler(cudaStatus, "cudaDeviceReset failed!");

    return 0;
}

cudaError_t NavierStokesCuda(float *c, const float *a, const float *b, unsigned int size)
{
    float* dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
    cudaError_t cudaStatus;

    vector<reference_wrapper<float*>> bidoof;
    bidoof.insert(bidoof.end(), { dev_c, dev_a, dev_b });

    cudaStatus = cudaSetDevice(0); //Assumes no multi-GPU
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    CudaMemoryAllocator(bidoof, (size_t) size, sizeof(float));
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = CopyFunction("cudaMemcpy failed!", dev_a, a, cudaMemcpyHostToDevice, cudaStatus, (size_t)size, sizeof(float));
    cudaStatus = CopyFunction("cudaMemcpy failed!", dev_b, b, cudaMemcpyHostToDevice, cudaStatus, (size_t)size, sizeof(float));

    // Launch a kernel on the GPU with one thread for each element.
    VelocityKernel<<<1, size>>>(dev_c, dev_a, dev_b);

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
    cudaStatus = CopyFunction("cudaMemcpy failed!", c, dev_c, cudaMemcpyDeviceToHost, cudaStatus, (size_t)size, sizeof(float));
    if (cudaStatus != cudaSuccess) {
        CudaMemoryFreer(bidoof);
    }
    return cudaStatus;
}