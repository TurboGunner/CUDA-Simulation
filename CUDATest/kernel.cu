﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "fluid_sim_cuda.cuh"

#include "handler_methods.hpp"
#include "vector_field.hpp"
#include "fluid_sim.hpp"

//#include "gui_driver.hpp"

#include <iostream>
#include <functional>

#include <stdio.h>

using std::vector;
using std::reference_wrapper;
using std::function;

int WinMain(int argc, char* argv[]) {
    const float FPS_10 = .1f, FPS_24 = 0.04166f, FPS_60 = 0.0166f;

    unsigned int iter = 32, side_bound = 128;
    uint3 sim_dimensions;

    sim_dimensions.x = side_bound;
    sim_dimensions.y = side_bound;
    sim_dimensions.z = side_bound;

    FluidSim simulation(FPS_60, 1.0f, 0.6f, sim_dimensions, iter, 0.332f, SimMethod::LBM);

    cudaError_t cuda_status = cudaSuccess;

    function<cudaError_t()> set_device_func = []() { return cudaSetDevice(0); };
    cuda_status = WrapperFunction(set_device_func, "cudaSetDevice failed!", "main",
        cuda_status, "Do you have a CUDA-capable GPU installed?");

    simulation.Simulate();

    cuda_status = cudaDeviceReset();
    CudaExceptionHandler(cuda_status, "cudaDeviceReset failed!");

    return 0;
}