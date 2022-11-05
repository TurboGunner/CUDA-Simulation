#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "matrix.cuh"

#include <math.h>

class Vector3D {
public:
	__host__ __device__ Vector3D() = default;

	__host__ __device__ Vector3D(float x_in, float y_in, float z_in);

	__host__ __device__  void operator=(const Vector3D& vector);

	__host__ __device__ inline float x() const {
		return dim[0];
	}

	__host__ __device__ inline float y() const {
		return dim[1];
	}

	__host__ __device__ inline float z() const {
		return dim[2];
	}

	__host__ __device__ Vector3D operator+(const float t);

	__host__ __device__ Vector3D operator-(const float t);

	__host__ __device__ Vector3D operator*(const float t);

	__host__ __device__ Vector3D operator/(const float t);

	__host__ __device__ Vector3D operator+(const Vector3D& vector);

	__host__ __device__ Vector3D operator-(const Vector3D& vector);

	__host__ __device__ Vector3D operator*(const Vector3D& vector);

	__host__ __device__ Vector3D& operator*=(const float t);

	__host__ __device__ Vector3D operator/(const Vector3D& vector);

	__host__ __device__ Vector3D& operator+=(const Vector3D& vector);

	__host__ __device__ Vector3D& operator/=(const Vector3D& vector);

	__host__ __device__ Vector3D& operator/=(const float t);

	__host__ __device__ inline Vector3D Squared() {
		return Vector3D(dim[0] * dim[0], dim[1] * dim[1], dim[2] * dim[2]);
	}

	__host__ __device__ inline Vector3D Negative() {
		return Vector3D(-dim[0], -dim[1], -dim[2]);
	}

	__host__ __device__ inline float SquaredLength(const Vector3D& vector1) {
		return vector1.dim[0] * vector1.dim[0] + vector1.dim[1] * vector1.dim[1] + vector1.dim[2] * vector1.dim[2];
	}

	__host__ __device__ inline float Length(const Vector3D& vector1) {
		return sqrt(vector1.dim[0] * vector1.dim[0] + vector1.dim[1] * vector1.dim[1] + vector1.dim[2] * vector1.dim[2]);
	}

	__host__ __device__ inline Vector3D UnitVector(const Vector3D& vector) {
		Vector3D output = vector;
		return output / Length(vector);
	}

	__host__ __device__ inline float DotProduct(const Vector3D& vector1, const Vector3D& vector2) {
		return (vector1.dim[0] * vector2.dim[0]) + (vector1.dim[1] * vector2.dim[1]) + (vector1.dim[2] * vector2.dim[2]);
	}

	__host__ __device__ inline Vector3D CrossProduct(const Vector3D& vector1, const Vector3D& vector2) {
		return Vector3D((vector1.dim[1] * vector2.dim[2] - vector1.dim[2] * vector2.dim[1]),
			(-(vector1.dim[0] * vector2.dim[2] - vector1.dim[2] * vector2.dim[0])),
			(vector1.dim[0] * vector2.dim[1] - vector1.dim[1] * vector2.dim[0]));
	}

	__host__ __device__ inline Vector3D Truncate() {
		return Vector3D((int)dim[0], (int)dim[1], (int)dim[2]);
	}

	__host__ __device__ Matrix ToMatrix();

	__host__ __device__ void ToMatrix(Matrix* matrix);

	__host__ __device__ void Reset();

	__host__ __device__ Vector3D Clamp(const float min, const float max);

	float dim[3];
};