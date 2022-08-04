#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "raypath.cuh"

#include <stdio.h>

#include <functional>
#include <iostream>

using std::function;

int main() {
    cudaError_t cuda_status;

    function<cudaError_t()> set_device_func = []() { return cudaSetDevice(0); };
    cuda_status = WrapperFunction(set_device_func, "cudaSetDevice failed!", "main",
        cuda_status, "Do you have a CUDA-capable GPU installed?");

    uint2 size;
    size.x = 200;
    size.y = 200;

    Vector3D* frame_buffer = AllocateTexture(size, cuda_status);

    std::cout << "P3\n" << size.x << " " << size.y << "\n255\n";
    for (int j = size.y - 1; j >= 0; j--) {
        for (int i = 0; i < size.x; i++) {
            size_t pixel_index = j * size.x + i;
            int ir = int(255.99 * fb[pixel_index].r());
            int ig = int(255.99 * fb[pixel_index].g());
            int ib = int(255.99 * fb[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    cuda_status = cudaDeviceReset();
    CudaExceptionHandler(cuda_status, "cudaDeviceReset failed!");
    return 0;
}
