#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "raypath.cuh"

#include <iostream>
#include <stdexcept>

class MaterialData {
public:
    __host__ __device__ MaterialData(int init_size) {
        table_size_ = init_size;
        cudaMalloc(&table_, (size_t)sizeof(Hitable) * table_size_ * BUFFER_MULTIPLIER_);
        cudaMallocHost(&table_host_, sizeof(Hitable) * table_size_ * BUFFER_MULTIPLIER_);
    }

    __host__ __device__ void* operator new(size_t size) {
        void* ptr;
        ptr = malloc(sizeof(MaterialData));
        return ptr;
    }

	void DeviceTransfer(cudaError_t& cuda_status, MaterialData*& src, MaterialData*& ptr) {
        cuda_status = CopyFunction("DeviceTransferTable", table_, table_host_, cudaMemcpyHostToDevice, cuda_status, sizeof(Hitable), table_size_);
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
		cuda_status = CopyFunction("HostTransferTable", table_host_, table_, cudaMemcpyDeviceToHost, cuda_status, sizeof(Hitable), table_size_);
		cudaDeviceSynchronize();
	}

    __host__ __device__ Hitable& Get(const int& index) {
#ifdef __CUDA_ARCH__
        return table_[index];
#else
        if (index < 0) {
            throw std::invalid_argument("Invalid index!");
        }
        return table_host_[index];
#endif
    }

    __host__ __device__ Hitable* Get() {
#ifdef __CUDA_ARCH__
        return table_;
#else
        return table_host_;
#endif
    }

    __host__ __device__ void Put(const int& key, const Hitable& value) {

        if (key < 0 || key >= table_size_ * BUFFER_MULTIPLIER_) {
            return;
        }
#ifdef __CUDA_ARCH__
        printf("%f", key.IX(cbrt(hash_table_size_)));
        table_[key.IX(cbrt(hash_table_size_))] = value;
#else
        table_host_[key] = value;
#endif
    }

    MaterialData* device_alloc_ = nullptr;

	Hitable* table_, *table_host_;
	size_t table_size_;

    bool device_allocated_status = false;

    const unsigned int BUFFER_MULTIPLIER_ = 2;
};