#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <stdexcept>
#include "../CUDATest/handler_methods.hpp"

class Matrix {
public:
	__host__ __device__ Matrix() = default;

	__host__ __device__ Matrix(const size_t& rows_in, const size_t& columns_in, const bool& local = false);

	__host__ __device__ static Matrix* Create(const size_t& rows, const size_t& columns, const bool& local = false);

	__host__ __device__ size_t IX(size_t row, size_t column) const;

	__host__ __device__ Matrix Transpose();

	__host__ __device__ float& Get(const int& index);

	__host__ __device__ float& Get(const size_t& row, const size_t& column);

	__host__ __device__ float& operator[](const int& index);

	__host__ __device__ void Set(const float& value, const int& index);
	__host__ __device__ void Set(const float& value, const size_t& row, const size_t& column);

    __host__ __device__  void GetCofactor(Matrix& output_matrix, int p, int q, int n);

    __host__ __device__ static int Determinant(Matrix& matrix, size_t length);

    __host__ __device__ static void Adjoint(Matrix& matrix, Matrix& adjoint);

    __host__ __device__ static bool Inverse(Matrix& matrix, Matrix& inverse);

    Matrix* device_alloc;

	size_t rows, columns;

private:
	float* data, *data_device;

    bool is_square = true;
};