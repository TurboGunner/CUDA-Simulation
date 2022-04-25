﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "fluid_sim_cuda.cuh"

#include "handler_methods.hpp"
#include "vector_field.hpp"
#include "fluid_sim.hpp"

#include <iostream>
#include <functional>

#include <stdio.h>

using std::vector;
using std::reference_wrapper;
using std::function;

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


    function<cudaError_t()> set_device_func = []() { return cudaSetDevice(0); };
    cuda_status = WrapperFunction(set_device_func, "cudaSetDevice failed!", "main", 
        cuda_status, "Do you have a CUDA-capable GPU installed?");

    float a_fac = simulation.dt_ * simulation.diffusion_ * (simulation.size_x_ - 2) * (simulation.size_x_ - 2);
    float c_fac = 1.0f + (4.0f * a_fac);

    std::cout << simulation.density_.ToString() << std::endl;

    //CudaExceptionHandler(cuda_status, "LinearSolverCuda failed!");

    cuda_status = cudaDeviceReset();
    CudaExceptionHandler(cuda_status, "cudaDeviceReset failed!");

    return 0;
}