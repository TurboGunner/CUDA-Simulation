#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../CUDATest/handler_classes.hpp"
#include "../CUDATest/handler_methods.hpp"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/iterator/counting_iterator.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <assert.h>
#include <stdio.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using std::string;
using std::vector;

class Matrix {
public:
	__host__ __device__ Matrix() = default;

	__host__ __device__ Matrix(const size_t rows_in, const size_t columns_in, const bool local = false, const bool host_in = true);

	__host__ __device__ static Matrix* Create(const size_t rows, const size_t columns, const bool local = false, const bool host = false);

	__host__ cudaError_t Destroy();

	__host__ static void DeleteAllocations(const vector<Matrix*>& matrices);

	__host__ static Matrix* MatrixMassAllocation(const size_t size, const size_t rows, const size_t columns);

	__host__ __device__ static void CopyMatrixOnPointer(Matrix* matrix, Matrix& copy);

	__host__ __device__ size_t IX(size_t row, size_t column) const;

	__host__ __device__ Matrix Transpose();

	__host__ __device__ static Matrix* TransposeGPU(Matrix* matrix);
	__host__ __device__ static Matrix* MultiplyGPU(Matrix* matrix_A, Matrix* matrix_B);

	__host__ __device__ Matrix operator*(const float scalar);

	__host__ __device__ Matrix& operator*=(const float scalar);

	__host__ __device__ Matrix& operator+=(const Matrix& matrix);

	__host__ __device__ float& Get(const int index) const;
	__host__ __device__ float& Get(const size_t row, const size_t column) const;

	__host__ __device__ float& operator[](const int index);

	__host__ __device__ static Matrix* DiagonalMatrix(const float* points, const size_t row, const size_t column);

	__host__ __device__ static void PopulateDiagonalMatrix(Matrix* matrix, const float* points, const size_t row, const size_t column);

	__host__ __device__ float* Row(const size_t index);

	__host__ __device__ float* Column(const size_t index);

	__host__ __device__ void Set(const float value, const int index);
	__host__ __device__ void Set(const float value, const size_t row, const size_t column);

    __host__ __device__  void GetCofactor(Matrix& output_matrix, int p, int q, int n);

    __host__ __device__ static float Determinant(Matrix& matrix, size_t length);

    __host__ __device__ static void Adjoint(Matrix& matrix, Matrix& adjoint);

	__host__ __device__ static bool Inverse(Matrix& matrix, Matrix& inverse);

	__host__ static Matrix* GMatrixTerm(Matrix* matrix, Matrix* matrix_T, Matrix* weights);

	__host__ static Matrix* Weights(Matrix* matrix);

	__host__ static vector<Matrix*> WeightedLeastSquares(Matrix* matrix);

	__host__ __device__ Matrix AbsoluteValue();

	__host__ __device__ Matrix Reciprocal();

	__host__ cudaError_t DeviceTransfer(Matrix* src);

	__host__ cudaError_t HostTransfer();

	__host__ string ToString(const char* label = nullptr);

	__host__ __device__ void PrintMatrix(const char* label = nullptr);

	__host__ static cudaError_t PopulateRandomHost(Matrix* matrix, const float min, const float max);

	__host__ __device__ static void AddOnPointer(Matrix* matrix, Matrix add);

	__host__ __device__ static void MultiplyScalarOnPointer(Matrix* matrix, const float multi);

	__host__ __device__ void operator=(const Matrix& copy) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				Get(j, i) += copy.Get(j, i);
			}
		}

		rows = copy.rows;
		columns = copy.columns;
	}

    Matrix* device_alloc;

	size_t rows, columns;

	float* Data(bool device_mode = true) const {
		if (device_mode) {
			return data_device;
		}
		return data;
	}

private:
	float* data, *data_device;

    bool is_square = true;

	bool device_allocated_status = false, local = false, host = false;
};

__global__ void TransposeKernel(Matrix* matrix, Matrix* output);

__global__ void MultiplyKernel(Matrix* matrix_A, Matrix* matrix_B, Matrix* output);