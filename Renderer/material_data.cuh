#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "raypath.cuh"

#include <iostream>
#include <stdexcept>

#include "../CUDATest/handler_methods.hpp"

const static unsigned int DEFAULT_SIZE = 1;

const static unsigned int BUFFER_MULTIPLIER_ = 2;

class MaterialData : public Hitable {
public:

    __host__ __device__ MaterialData() { }

    __host__ __device__ MaterialData(int init_size) {
        InitializeArray(init_size);
    }

    __host__ __device__ ~MaterialData() {
        cudaFree(table_);
        free(table_host_);

        cudaFree(device_alloc_);
    }

    __host__ __device__ void InitializeArray(int init_size) {
        table_size_ = init_size;
        cudaMalloc(&table_, sizeof(Hitable*) * table_size_ * BUFFER_MULTIPLIER_);
        cudaMallocHost(&table_host_, sizeof(Hitable*) * table_size_ * BUFFER_MULTIPLIER_);
        is_initialized = true;
    }

    __host__ __device__ void* operator new(size_t size) {
        void* ptr;
        cudaMallocHost(&ptr, sizeof(MaterialData));
        return ptr;
    }

    __host__ __device__ void operator delete(void* ptr) {
        free(ptr);
    }

    __host__ void DeviceTransfer(cudaError_t& cuda_status, MaterialData*& src, MaterialData*& ptr) {
        cuda_status = CopyFunction("DeviceTransferTable", table_, table_host_, cudaMemcpyHostToDevice, cuda_status, sizeof(Hitable*), table_size_);
        if (!device_allocated_status) {
            cuda_status = cudaMalloc(&ptr, sizeof(MaterialData));
            device_allocated_status = true;
            cuda_status = CopyFunction("DeviceTransferObject", ptr, src, cudaMemcpyHostToDevice, cuda_status, sizeof(MaterialData), 1);
            device_alloc_ = ptr;
        }
        else {
            ptr = device_alloc_;
        }
    }

    __host__ void HostTransfer(cudaError_t& cuda_status) {
        cuda_status = CopyFunction("HostTransferTable", table_host_, table_, cudaMemcpyDeviceToHost, cuda_status, sizeof(Hitable*), table_size_);
        cudaDeviceSynchronize();
    }

    __host__ __device__ Hitable* Get(const int& index) {
#ifdef __CUDA_ARCH__
        return table_[index];
#else
        if (index < 0) {
            throw std::invalid_argument("Invalid index!");
        }
        return table_host_[index];
#endif
    }

    __host__ __device__ Hitable** Get() {
#ifdef __CUDA_ARCH__
        return table_;
#else
        return table_host_;
#endif
    }

    __host__ __device__ void Put(const int& key, Hitable* value) {

        if (key < 0 || key >= table_size_ * BUFFER_MULTIPLIER_) {
            return;
        }
#ifdef __CUDA_ARCH__
        table_[key] = value;
#else
        table_host_[key] = value;
#endif
    }

    __device__ virtual bool Hit(const Ray& ray, float t_min, float t_max, RayHit& hit) const;

    MaterialData* device_alloc_ = nullptr;

    Hitable** table_, **table_host_;
    size_t table_size_;

private:
    bool is_initialized = false;
    bool device_allocated_status = false;
};

__device__ bool MaterialData::Hit(const Ray& ray, float t_min, float t_max, RayHit& hit) const {
    RayHit temp_hit;
    bool successful_hit = false;
    double closest_so_far = t_max;

    for (size_t i = 0; i < table_size_; i++) {
        if (table_[i]->Hit(ray, t_min, closest_so_far, temp_hit)) {
            successful_hit = true;
            closest_so_far = temp_hit.t;
            hit = temp_hit;
        }
    }
    return successful_hit;
}