#pragma once

#include "cuda_runtime.h"

#include <stdexcept>
#include <string>
#include <iostream>

using std::string;

template <typename K>
struct HashFunc {
    /// <summary> Default provided hashing function for this data structure. 
    /// <para> Should be replaced depending on use-case and key input type. </para> </summary>
    __host__ __device__ size_t operator()(const K& key, size_t size) const
    {
        size_t hash = (size_t)(key);
        return (hash << 1) % size;
    }
};

template <typename K, typename V, typename F = HashFunc<K>>
class HashMap {
public:
    /// <summary> The default constructor for the HashMap. </summary>
    HashMap() {
        hash_table_size_ = DEFAULT_SIZE;
        Initialization();
    }

    /// <summary> Loaded Constructor, allocates with size argument. </summary>
    HashMap(const size_t& hash_table_size) {
        if (hash_table_size < 1) {
            throw std::invalid_argument("The input size for the hash table should be at least 1!");
        }
        hash_table_size_ = hash_table_size;
        Initialization();
    }

    /// <summary> Helper method to initialize unified memory with the values and the accompanying hash map. </summary>
    void Initialization() {
        cudaMallocManaged(&table_, (size_t)sizeof(V) * hash_table_size_);
        cudaMallocManaged(&hashes_, (size_t)sizeof(int) * hash_table_size_);
    }

    /// <summary> Destructor for the HashMap implementation. </summary>
    ~HashMap() {
        cudaFree(table_);
        cudaFree(hashes_);
    }

    /// <summary> Allocates new HashMap pointer, new keyword overload. </summary>
    void* operator new(size_t size) {
        void* ptr;
        cudaMallocManaged(&ptr, sizeof(HashMap<K, V, HashFunc<K>>)); //Allocates the size of the 
        cudaDeviceSynchronize();
        return ptr;
    }

    /// <summary> Deallocates HashMap pointers, delete keyword overload. </summary>
    void operator delete(void* ptr) {
        cudaDeviceSynchronize();
        cudaFree(ptr);
    }

    /// <summary> Associative array logic, allows for the mapping of hashes to index positions. </summary>
    __host__ __device__ long FindHash(const int& hash) {
        if (hash > hash_table_size_ || hashes_[hash] == 0) {
            return -1;
        }
        return hashes_[hash] - 1;
    }

    /// <summary> Accessor method when key is an input. </summary>
    __host__ __device__ V Get(const K& key) {
        size_t hash = hash_func_(key, hash_table_size_);
        long hash_pos = FindHash(hash);
        if (hash_pos == -1) {
            printf("%s", "Invalid Index!\n");
        }
        return table_[hash_pos];

    }

    /// <summary> Accessor method when an integer index is an input. </summary>
    __host__ __device__ V Get(const int& index) {
#ifdef __CUDA_ARCH__
        if (index < 0 || index >= size_) {
            printf("%s", "Invalid Index!\n");
        }
#else
        if (index < 0 || index >= size_) {
            throw std::invalid_argument("Invalid index!");
        }
#endif
        return table_[index];
    }

    /// <summary> Accessor method when int index is an input. </summary>
    __host__ void Put(const K& key, const V& value) {
        size_t hash = hash_func_(key, hash_table_size_);
        long hash_pos = FindHash(hash);
        if (hash_pos == -1) {
            hashes_[hash] = size_ + 1;

            table_[size_] = value;
            size_++;
        }
        else {
            table_[hash_pos] = value;
        }
    }

    /// <summary> Removes hash table value, treated as erased in the pointers logically. </summary>
    __host__ __device__ void Remove(const K& key) {
        size_t hash = hash_func_(key, hash_table_size_);
        long hash_pos = FindHash(hash);
        if (hash_pos == -1) {
            return;
        }

        table_[hash_pos] = 0;
        hashes_[hash] = -1;

        size_--;
    }

    /// <summary> Does ToString output. Host only. </summary>
    __host__ string ToString() {
        string output;
        for (size_t i = 0; i < size_; i++) {
            output += std::to_string(table_[i]);
        }
        return output;
    }

    /// <summary> Calls get from operator overload based on the key input. </summary>
    __host__ __device__ V& operator[](const K& key) {
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
        if (src == *this) {
            return *this;
        }
        table_ = src.table_;
        return *this;
    }

    long size_ = 0;
    size_t hash_table_size_;

private:
    V* table_;
    int* hashes_;

    F hash_func_;

    const size_t DEFAULT_SIZE = 10000;
};