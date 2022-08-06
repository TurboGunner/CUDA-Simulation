﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "raypath.cuh"

#include <stdio.h>

#include <CImg.h>

#include <functional>
#include <iostream>

using std::function;

using namespace cimg_library;

int main() {
    cudaError_t cuda_status = cudaSuccess;

    function<cudaError_t()> set_device_func = []() { return cudaSetDevice(0); };
    cuda_status = WrapperFunction(set_device_func, "cudaSetDevice failed!", "main",
        cuda_status, "Do you have a CUDA-capable GPU installed?");

    uint2 size;
    size.x = 1024;
    size.y = 1024;

    Vector3D* frame_buffer = AllocateTexture(size, cuda_status);

    cudaDeviceSynchronize();

    CImg<float> img(size.x, size.y, 1, 3, 0);
    cimg_forXY(img, x, y) {
        size_t pixel_index = y * size.x + x;
        Vector3D point = frame_buffer[pixel_index];

        img(x, y, 0) = point.x() * 255.99f;
        img(x, y, 1) = point.y() * 255.99f;
        img(x, y, 2) = point.z() * 255.99f;
    }

    img.save_bmp("filename.bmp");

    cuda_status = cudaDeviceReset();
    CudaExceptionHandler(cuda_status, "cudaDeviceReset failed!");
    return 0;
}
