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

void SimulationOperations(FluidSim& simulation) {
    simulation.AddVelocity(IndexPair(0, 0), 10, 10);
    simulation.AddVelocity(IndexPair(1, 0), 22, 2);
    simulation.AddVelocity(IndexPair(1, 1), 2022, 220);

    simulation.AddDensity(IndexPair(1, 1), 10.0f);
    simulation.AddDensity(IndexPair(2, 2), 10.0f);

    simulation.Diffuse(1, 1.0f, 1.0f);
    simulation.Project();
    simulation.Advect(0, 2.0f);
    //std::cout << simulation.velocity_.ToString() << std::endl;
}

int main()
{
    unsigned int iter = 8, side_bound = 8;
    FluidSim simulation(.1f, 1.0f, 1, side_bound, side_bound, iter);
    SimulationOperations(simulation);

    cudaError_t cuda_status = cudaSuccess;

    float a_fac = simulation.dt_ * simulation.diffusion_ * (simulation.size_x_ - 2) * (simulation.size_x_ - 2);
    float c_fac = 1.0f + (4.0f * a_fac);

    std::cout << simulation.density_.ToString() << std::endl;

    //CudaExceptionHandler(cuda_status, "LinearSolverCuda failed!");

    cuda_status = cudaDeviceReset();
    CudaExceptionHandler(cuda_status, "cudaDeviceReset failed!");

    return 0;
}

cudaError_t NavierStokesCuda(float *c, const float *a, const float *b, unsigned int size)
{
    float* dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
    cudaError_t cuda_status;

    vector<reference_wrapper<float*>> bidoof;
    bidoof.insert(bidoof.end(), { dev_c, dev_a, dev_b });

    cuda_status = cudaSetDevice(0); //Assumes no multi-GPU
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        return cuda_status;
    }

    CudaMemoryAllocator(bidoof, (size_t) size, sizeof(float));
    // Copy input vectors from host memory to GPU buffers.
    cuda_status = CopyFunction("cudaMemcpy failed!", dev_a, a, cudaMemcpyHostToDevice, cuda_status, (size_t)size, sizeof(float));
    cuda_status = CopyFunction("cudaMemcpy failed!", dev_b, b, cudaMemcpyHostToDevice, cuda_status, (size_t)size, sizeof(float));

    // Launch a kernel on the GPU with one thread for each element.

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cuda_status));
        CudaMemoryFreer(bidoof);
        return cuda_status;
    }
    
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cuda_status);
        CudaMemoryFreer(bidoof);
        return cuda_status;
    }

    // Copy output vector from GPU buffer to host memory.
    cuda_status = CopyFunction("cudaMemcpy failed!", c, dev_c, cudaMemcpyDeviceToHost, cuda_status, (size_t)size, sizeof(float));
    if (cuda_status != cudaSuccess) {
        CudaMemoryFreer(bidoof);
    }
    return cuda_status;
}