#pragma once

#include "cuda_runtime.h"

#include "index_pair_cuda.cuh"

#include "handler_methods.hpp"

#include <stdexcept>
#include <string>
#include <iostream>
#include <functional>
#include <vector>

using std::string;
using std::reference_wrapper;
using std::vector;

template <typename V>
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
        printf("%zu\n", hash_table_size_);
        Initialization();
    }

    /// <summary> Helper method to initialize unified memory with the values and the accompanying hash map. </summary>
    __host__ __device__ void Initialization() {
        cudaMalloc(&table_, (size_t)sizeof(V) * hash_table_size_);
        cudaMalloc(&hashes_, (size_t)sizeof(int) * hash_table_size_);

        table_host_ = new V[hash_table_size_];
        hashes_host_ = new int[hash_table_size_];

        for (size_t i = 0; i < hash_table_size_; i++) {
            hashes_host_[i] = 0;
        }
        printf("%s", "HashMap constructor instantiated!\n");
    }

    __host__ __device__ size_t hash_func_(IndexPair i1) {
        size_t hash = i1.x + (i1.y * (sqrt(hash_table_size_)));
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
#ifdef __CUDA_ARCH__
#else
        printf("%u\n", size);
#endif
        ptr = malloc(sizeof(HashMap<V>));
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
    __host__ __device__ V& Get(const IndexPair& key) {
        size_t hash = hash_func_(key);
        long hash_pos = FindHash(hash);
        if (hash_pos == -1) {
            //printf("Hash: %u ", hash);
        }
#ifdef __CUDA_ARCH__
        return table_[hash_pos];
#else
        return table_host_[hash_pos];
#endif
    }

    /// <summary> Accessor method when an integer index is an input. </summary>
    __host__ __device__ V& Get(const int& index) {
#ifdef __CUDA_ARCH__
        if (index < 0 || index >= size_) {
            //printf("%s", "Invalid Index!\n");
        }
        return table_[index];
#else
        if (index < 0) {
            throw std::invalid_argument("Invalid index!");
        }
        return table_host_[index];
#endif
    }

    /// <summary> Accessor method when int index is an input. </summary>
    __host__ __device__ void Put(const IndexPair& key, const V& value) {
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
        printf("%f", key.IX(sqrt(hash_table_size_)));
         table_[key.IX(sqrt(hash_table_size_))] = value;
#else
        table_host_[hash_pos] = value;
#endif
    }

    /// <summary> Accessor method when int index is an input. </summary>
    __host__ __device__ void Put(int key, V value) {
#ifdef __CUDA_ARCH__
            table_[key] = value;
#else
            table_host_[key] = value;
#endif
    }

    __host__ __device__ size_t Size() const {
        return hash_table_size_;
    }
    void DeviceTransfer(cudaError_t& cuda_status, HashMap<V>*& src, HashMap<V>*& ptr) {
        cuda_status = CopyFunction("DeviceTransferTable", table_, table_host_, cudaMemcpyHostToDevice, cuda_status, sizeof(V), hash_table_size_);
        cuda_status = CopyFunction("DeviceTransferHash", hashes_, hashes_host_, cudaMemcpyHostToDevice, cuda_status, sizeof(int), hash_table_size_);
        cuda_status = cudaMalloc(&ptr, sizeof(HashMap<V>));
        cuda_status = CopyFunction("DeviceTransferObject", ptr, src, cudaMemcpyHostToDevice, cuda_status, sizeof(HashMap<V>), 1);
        device_alloc_ = ptr;
    }

    __host__ void HostTransfer(cudaError_t& cuda_status) {
        cuda_status = CopyFunction("HostTransferTable", table_host_, table_, cudaMemcpyDeviceToHost, cuda_status, sizeof(V), hash_table_size_);
        cudaFree(device_alloc_);
    }

    /// <summary> Removes hash table value, treated as erased in the pointers logically. </summary>
    __host__ __device__ void Remove(const IndexPair& key) {
        size_t hash = hash_func_(key, hash_table_size_);
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

    /// <summary> Does ToString output. Host only. </summary>
    __host__ string ToString() {
        string output;
        for (size_t i = 0; i < size_; i++) {
            output += table_host_[i] + "\n";
        }
        return output;
    }

    /// <summary> Calls get from operator overload based on the key input. </summary>
    __host__ __device__ V& operator[](const IndexPair& key) {
        V output = Get(key);
        return output;
    }

    /// <summary> Calls get from operator overload based on the integer index input. </summary>
    __host__ __device__ V& operator[](const int& index) {
        V output = Get(index);
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

private:
    V* table_, *table_host_;
    int* hashes_, *hashes_host_;

    const size_t DEFAULT_SIZE = 10000;

    HashMap<V>* device_alloc_ = nullptr;

    long size_ = 0;
    size_t hash_table_size_;
};

/**
Notes:
1) Maybe use a vector in order to make it dynamically resizable
2) Maybe use this as a means to have it keep track of allocations and subsequently add more memory
3) Maybe create table of pointers
*/