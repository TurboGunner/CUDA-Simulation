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
        unsigned long hash = (unsigned long)(key) % size;
        return hash;
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
        cudaMallocManaged(&table_, (size_t)sizeof(V) * hash_table_size_);
        cudaMallocManaged(&hashes_, (size_t)sizeof(int) * hash_table_size_);
        printf("%s", "HashMap constructor instantiated!\n");
    }

    /// <summary> Destructor for the HashMap implementation. </summary>
    ~HashMap() {
        cudaFree(table_);
        cudaFree(hashes_);
    }

    /// <summary> Allocates new HashMap pointer, new keyword overload. </summary>
    __host__ __device__ void* operator new(size_t size) {
        void* ptr;
#ifdef __CUDA_ARCH__
        printf("%u\n", size);
#endif
        cudaMallocManaged(&ptr, sizeof(HashMap<K, V, HashFunc<K>>));
        //cudaDeviceSynchronize();
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
        return hashes_[hash] - 1;
    }

    /// <summary> Accessor method when key is an input. </summary>
    __host__ __device__ V Get(const K& key) {
        size_t hash = hash_func_(key, hash_table_size_);
        long hash_pos = FindHash(hash);
        if (hash_pos == -1) {
            printf("Hash: %u ", hash);
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
        if (hash_table_size_ - 1 == size_) {
            printf("%zu\n", size_);
        }
        size_t hash = hash_func_(key, hash_table_size_);
        long hash_pos = FindHash(hash);
        if (hash_pos == -1 && size_ <= hash_table_size_) {
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
            output += table_[i] + "\n";
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
        if (table_ == src.table_) {
            return *this;
        }
        Initialization();
        *(table_) = *(src.table_);
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