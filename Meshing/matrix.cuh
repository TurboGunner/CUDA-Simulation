#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../CUDATest/handler_classes.hpp"
#include "../CUDATest/handler_methods.hpp"

#include <assert.h>
#include <stdio.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using std::string;
using std::vector;

template<size_t rows, size_t columns>
class Matrix {
public:
	__host__ __device__ Matrix() = default;

	__host__ __device__ size_t IX(const size_t row, const size_t column) const {
		return column + (rows * row);
	}

	__host__ __device__ float Get(const int index) const {
		assert(index < rows * columns && index >= 0);
		return data[index];
	}

	__host__ __device__ float Get(const size_t row, const size_t column) const {
		return Get(IX(row, column));
	}

	__host__ __device__ float operator[](const int index) {
		return Get(index);
	}

	__host__ __device__ void CopyMatrixOnPointer(Matrix* matrix, Matrix& copy) {
		for (size_t i = 0; i < matrix->Rows(); i++) {
			for (size_t j = 0; j < matrix->Columns(); j++) {
				matrix->Set(copy.Get(j, i), j, i);
			}
		}
	}

	__host__ __device__ Matrix Transpose() {
		Matrix output;
		for (size_t i = 0; i < columns; i++) {
			for (size_t j = 0; j < rows; j++) {
				output.Set(Get(j, i), i, j);
			}
		}
		return output;
	}

	__host__ __device__ Matrix AbsoluteValue() {
		Matrix<rows, columns> output;

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				output.Set(abs(Get(i, j)), i, j);
			}
		}

		return output;
	}

	__host__ __device__ Matrix Reciprocal() {
		Matrix<rows, columns> output;

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				float value_idx = Get(i, j);
				float evaluated = value_idx != 0 ? (1.0f / (value_idx)) : 0.0f;
				output.Set(evaluated, i, j);
			}
		}

		return output;
	}

	__host__ __device__ Matrix operator*(const float scalar) {
		Matrix<rows, columns> matrix;

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				matrix.Set(Get(j, i) * scalar, j, i);
			}
		}
		return matrix;
	}

	__host__ __device__ Matrix& operator*=(const float scalar) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				data[IX(j, i)] *= scalar;
			}
		}
		return *this;
	}

	__host__ __device__ Matrix& operator+=(const Matrix& matrix) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				data[IX(j, i)] += matrix.Get(j, i);
			}
		}
		return *this;
	}

	__host__ __device__ Matrix DiagonalMatrix(const float* points, const size_t rows, const size_t columns) {
		Matrix output;
		for (int i = 0; i < rows; i++) {
			output.Set(points[i], i, i);
		}

		return output;
	}

	__host__ __device__ static void PopulateDiagonalMatrix(Matrix matrix, const float points[], const size_t row, const size_t column) {
		for (int i = 0; i < matrix.Rows(); i++) {
			matrix.Set(points[i], i, i);
		}
	}

	__host__ __device__ float* Row(const size_t index) {
		float* output = (float*) malloc(columns * sizeof(float)); //Maybe use memcpy later?
		for (int i = 0; i < columns; i++) {
			output[i] = Get(i, index);
		}
		return output;
	}

	__host__ __device__ float* Column(const size_t index) {
		float* output = (float*) malloc(rows * sizeof(float)); //Maybe use memcpy later?
		for (int i = 0; i < rows; i++) {
			output[i] = Get(index, i);
		}
		return output;
	}

	__host__ __device__ void Set(const float value, const int index) {
		assert(index < rows* columns && index >= 0);
		data[index] = value;
	}

	__host__ __device__ void Set(const float value, const size_t row, const size_t column) {
		Set(value, IX(row, column));
	}


	__host__ string ToString(const char* label = nullptr) {
		string output;
		if (label) { // Nullptr default check
			output += label;
		}
		else {
			output += "\n\n";
		}
		for (size_t i = 0; i < rows; i++) {
			output += "\n";
			for (size_t j = 0; j < columns; j++) {
				output += " " + std::to_string(Get(IX(j, i)));
			}
		}
		return output;
	}

	__host__ __device__ void PrintMatrix(const char* label = nullptr) {
		if (label) { // Nullptr default check
			printf("\n\n%s", label);
		}
		else {
			printf("\n\n");
		}
		for (size_t i = 0; i < rows; i++) {
			printf("\n");
			for (size_t j = 0; j < columns; j++) {
				printf("%f ", Get(IX(j, i)));
			}
		}
	}

	__host__ __device__ static void AddOnPointer(Matrix* matrix, Matrix add) {
		*matrix += add;
	}

	__host__ __device__ static void MultiplyScalarOnPointer(Matrix* matrix, const float multi) {
		*matrix *= multi; //NOTE
	}

	__host__ __device__ const size_t Rows() const {
		return rows;
	}

	__host__ __device__ const size_t Columns() const {
		return columns;
	}

	__host__ __device__ const bool IsSquare() const {
		return rows == columns;
	}

	__host__ __device__ void operator=(const Matrix& copy) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				Set(copy.Get(j, i), j, i);
			}
		}
	}

private:
	float data[rows * columns];
};

template<size_t rows, size_t columns>
__global__ inline void TransposeKernel(Matrix<rows, columns>* matrix, Matrix<rows, columns>* output) {
	unsigned int y_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int x_bounds = blockIdx.y * blockDim.y + threadIdx.y;

	if (x_bounds <= output->rows && y_bounds <= output->columns) {
		output->Get(output->IX(threadIdx.y, threadIdx.x)) = matrix->Get(matrix->IX(threadIdx.x, threadIdx.y));
	}
}

template<size_t rows, size_t columns>
__global__ void MultiplyKernel(Matrix<rows, columns>* matrix_A, Matrix<rows, columns>* matrix_B, Matrix<rows, columns>* output);