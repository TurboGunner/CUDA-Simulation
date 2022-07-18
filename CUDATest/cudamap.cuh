#pragma once

#include "cuda_runtime.h"

#include "index_pair_cuda.cuh"

#include "handler_methods.hpp"

#include <stdexcept>
#include <string>
#include <iostream>

using std::string;

class HashMap {
public:
    /// <summary> The default constructor for the HashMap. </summary>
    HashMap() {
        hash_table_size_ = DEFAULT_SIZE;
        Initialization();
    }

    /// <summary> Loaded Constructor, allocates with size argument. </summary>
    __host__ __device__ HashMap(const size_t& hash_table_size) {
        if (hash_table_size < 1) {
            printf("%s\n", "The input size for the hash table should be at least 1!");
        }
        hash_table_size_ = hash_table_size;
        Initialization();
    }

    /// <summary> Helper method to initialize unified memory with the values and the accompanying hash map. </summary>
    __host__ __device__ void Initialization() {
        cudaMalloc(&table_, (size_t) sizeof(float) * hash_table_size_);
        cudaMalloc(&hashes_, (size_t) sizeof(int) * hash_table_size_);

        cudaMallocHost(&table_host_, sizeof(float) * hash_table_size_);
        cudaMallocHost(&hashes_host_, (size_t)sizeof(int) * hash_table_size_);

        for (size_t i = 0; i < hash_table_size_; i++) {
            hashes_host_[i] = 0;
        }
    }

    __host__ __device__ size_t hash_func_(IndexPair i1) {
        size_t hash = i1.x + (i1.y * cbrt(hash_table_size_)) + (i1.z * (cbrt(hash_table_size_) * cbrt(hash_table_size_)));
        return hash;
    }

    /// <summary> Destructor for the HashMap implementation. </summary>
    ~HashMap() {
        cudaFree(table_);
        free((void*) table_host_);
        cudaFree(hashes_);
        free((void*) hashes_host_);
    }

    /// <summary> Allocates new HashMap pointer, new keyword overload. </summary>
    __host__ __device__ void* operator new(size_t size) {
        void* ptr;
        ptr = malloc(sizeof(HashMap));
        return ptr;
    }

    /// <summary> Deallocates HashMap pointers, delete keyword overload. </summary>
    __host__ __device__ void operator delete(void* ptr) {
        cudaDeviceSynchronize();
        cudaFree(ptr);
    }

    /// <summary> Associative array logic, allows for the mapping of hashes to index positions. </summary>
    __host__ __device__ long FindHash(const int& hash) {
        if (hash > hash_table_size_) {
            return -1;
        }
#ifdef __CUDA_ARCH__
        return hashes_[hash] - 1;
#else
        return hashes_host_[hash] - 1;
#endif
    }

    /// <summary> Accessor method when key is an input. </summary>
    __host__ __device__ float& Get(const IndexPair& key) {
        size_t hash = hash_func_(key);
        long hash_pos = FindHash(hash);
#ifdef __CUDA_ARCH__
        return table_[hash_pos];
#else
        return table_host_[hash_pos];
#endif
    }

    /// <summary> Accessor method when an integer index is an input. </summary>
    __host__ __device__ float& Get(const int& index) {
#ifdef __CUDA_ARCH__
        return table_[index];
#else
        if (index < 0) {
            throw std::invalid_argument("Invalid index!");
        }
        return table_host_[index];
#endif
    }

    /// <summary> Accessor method when int index is an input. </summary>
    __host__ __device__ void Put(const IndexPair& key, const float& value) {
#ifdef __CUDA_ARCH__
#else
        size_t hash = hash_func_(key);
        long hash_pos = FindHash(hash);

        if (hash_pos == -1 && size_ <= hash_table_size_) {
            hashes_host_[hash] = size_ + 1;
            table_host_[size_] = value;
            size_++;
            return;
        }
#endif
#ifdef __CUDA_ARCH__
        printf("%f", key.IX(cbrt(hash_table_size_)));
         table_[key.IX(cbrt(hash_table_size_))] = value;
#else
        table_host_[hash_pos] = value;
#endif
    }

    /// <summary> Accessor method when int index is an input. </summary>
    __host__ __device__ void Put(int key, float value) {
#ifdef __CUDA_ARCH__
            table_[key] = value;
#else
            table_host_[key] = value;
#endif
    }

    /// <summary> Device Transfer method, uses a toggle boolean to ensure that the device pointer equivalent is only allocated once. </summary
    void DeviceTransfer(cudaError_t& cuda_status, HashMap*& src, HashMap*& ptr) {
        cuda_status = CopyFunction("DeviceTransferTable", table_, table_host_, cudaMemcpyHostToDevice, cuda_status, sizeof(float), hash_table_size_);
        cuda_status = CopyFunction("DeviceTransferHash", hashes_, hashes_host_, cudaMemcpyHostToDevice, cuda_status, sizeof(int), hash_table_size_);
        if (!device_allocated_status) {
            cuda_status = cudaMalloc(&ptr, sizeof(HashMap));
            device_allocated_status = true;
            cuda_status = CopyFunction("DeviceTransferObject", ptr, src, cudaMemcpyHostToDevice, cuda_status, sizeof(HashMap), 1);
            device_alloc_ = ptr;
        }
        else {
            ptr = device_alloc_;
        }
        std::cout << table_host_[0] << std::endl;
    }

    __host__ void HostTransfer(cudaError_t& cuda_status) {
        cuda_status = CopyFunction("HostTransferTable", table_host_, table_, cudaMemcpyDeviceToHost, cuda_status, sizeof(float), hash_table_size_);
        cudaDeviceSynchronize();
    }

    /// <summary> Removes hash table value, treated as erased in the pointers logically. </summary>
    __host__ __device__ void Remove(const IndexPair& key) {
        size_t hash = hash_func_(key);
        unsigned long hash_pos = FindHash(hash);
        if (hash_pos == -1) {
            return;
        }
        int hash_pos_empty = 0, hash_empty = -1;
#ifdef __CUDA_ARCH__
        table_[hash_pos] = hash_pos_empty;
        hashes_[hash] = hash_empty;
#else
        table_host_[hash_pos] = hash_pos_empty;
        hashes_host_[hash] = hash_empty;
#endif

        size_--;
    }

    /// <summary> Calls get from operator overload based on the key input. </summary>
    __host__ __device__ float& operator[](const IndexPair& key) {
        float output = Get(key);
        return output;
    }

    /// <summary> Calls get from operator overload based on the integer index input. </summary>
    __host__ __device__ float& operator[](const int& index) {
        float output = Get(index);
        return output;
    }

    /// <summary> Assignment operator overload, is a shallow copy of the input object. </summary>
    HashMap& operator=(const HashMap& src) {
        if (table_ == src.table_) {
            return *this;
        }
        table_ = src.table_;
        hashes_ = src.hashes_;
        return *this;
    }

    __host__ __device__ size_t Size() const {
        return hash_table_size_;
    }

    HashMap* device_alloc_ = nullptr;

private:
    float* table_, *table_host_;
    int* hashes_, *hashes_host_;

    long size_ = 0;
    size_t hash_table_size_;

    const size_t DEFAULT_SIZE = 20000;

    bool device_allocated_status = false; //Ensures that the device pointer equivalent is only allocated once
};