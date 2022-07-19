#pragma once

#include "cuda_runtime.h"

#include "index_pair_cuda.cuh"

#include "handler_methods.hpp"

#include <stdexcept>
#include <string>

using std::string;

class HashMap {
public:
    /// <summary> The default constructor for the HashMap. </summary>
    HashMap();

    /// <summary> Loaded Constructor, allocates with size argument. </summary>
    __host__ __device__ HashMap(const size_t& hash_table_size);

    /// <summary> Helper method to initialize unified memory with the values and the accompanying hash map. </summary>
    __host__ __device__ void Initialization();

    __host__ __device__ size_t hash_func_(IndexPair i1);

    /// <summary> Destructor for the HashMap implementation. </summary>
    ~HashMap();

    /// <summary> Allocates new HashMap pointer, new keyword overload. </summary>
    __host__ __device__ void* operator new(size_t size);

    /// <summary> Deallocates HashMap pointers, delete keyword overload. </summary>
    __host__ __device__ void operator delete(void* ptr);

    /// <summary> Associative array logic, allows for the mapping of hashes to index positions. </summary>
    __host__ __device__ long FindHash(const int& hash);

    /// <summary> Accessor method when key is an input. </summary>
    __host__ __device__ float& Get(const IndexPair& key);

    /// <summary> Accessor method when an integer index is an input. </summary>
    __host__ __device__ float& Get(const int& index);

    /// <summary> Accessor method when int index is an input. </summary>
    __host__ __device__ void Put(const IndexPair& key, const float& value);

    /// <summary> Accessor method when int index is an input. </summary>
    __host__ __device__ void Put(int key, float value);

    /// <summary> Device Transfer method, uses a toggle boolean to ensure that the device pointer equivalent is only allocated once. </summary>
    void DeviceTransfer(cudaError_t& cuda_status, HashMap*& src, HashMap*& ptr);

    __host__ void HostTransfer(cudaError_t& cuda_status);

    /// <summary> Removes hash table value, treated as erased in the pointers logically. </summary>
    __host__ __device__ void Remove(const IndexPair& key);

    /// <summary> Calls get from operator overload based on the key input. </summary>
    __host__ __device__ float& operator[](const IndexPair& key);

    /// <summary> Calls get from operator overload based on the integer index input. </summary>
    __host__ __device__ float& operator[](const int& index);

    /// <summary> Assignment operator overload, is a shallow copy of the input object. </summary>
    HashMap& operator=(const HashMap& src);

    /// <summary> Accessor for the hash table size. </summary>
    __host__ __device__ size_t Size() const;

    HashMap* device_alloc_ = nullptr;

private:
    float* table_, *table_host_;
    int* hashes_, *hashes_host_;

    long size_ = 0;
    size_t hash_table_size_;

    const size_t DEFAULT_SIZE = 20000;

    bool device_allocated_status = false; //Ensures that the device pointer equivalent is only allocated once
};