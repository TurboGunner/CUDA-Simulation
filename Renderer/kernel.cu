﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "raypath.cuh"

#include <stdio.h>

#include <functional>
#include <iostream>

using std::function;

int main() {
    cudaError_t cuda_status = cudaSuccess;

    function<cudaError_t()> set_device_func = []() { return cudaSetDevice(0); };
    cuda_status = WrapperFunction(set_device_func, "cudaSetDevice failed!", "main",
        cuda_status, "Do you have a CUDA-capable GPU installed?");

    uint2 size;
    size.x = 64;
    size.y = 64;

    Vector3D* frame_buffer = AllocateTexture(size, cuda_status);

    cudaDeviceSynchronize();

    std::cout << "P3\n" << size.x << " " << size.y << "\n255\n";
    for (int j = size.y - 1; j >= 0; j--) {
        for (int i = 0; i < size.x; i++) {
            size_t pixel_index = j * size.x + i;
            float ir = 255.99 * frame_buffer[pixel_index].x();
            float ig = 255.99 * frame_buffer[pixel_index].y();
            float ib = 255.99 * frame_buffer[pixel_index].z();
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    cuda_status = cudaDeviceReset();
    CudaExceptionHandler(cuda_status, "cudaDeviceReset failed!");
    return 0;
}
