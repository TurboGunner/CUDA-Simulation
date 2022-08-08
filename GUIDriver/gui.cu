
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../CUDATest/handler_methods.hpp"

#include "gui_driver.cuh"

#include <stdio.h>

#include <functional>

using std::function;

int WinMain(int argc, char* argv[]) {
    cudaError_t cuda_status = cudaSuccess;

    function<cudaError_t()> set_device_func = []() { return cudaSetDevice(0); };
    cuda_status = WrapperFunction(set_device_func, "cudaSetDevice failed!", "main",
        cuda_status, "Do you have a CUDA-capable GPU installed?");

    VulkanGUIDriver gui_driver;

    gui_driver.RunGUI();

    cuda_status = cudaDeviceReset();
    CudaExceptionHandler(cuda_status, "cudaDeviceReset failed!");

    return 0;
}