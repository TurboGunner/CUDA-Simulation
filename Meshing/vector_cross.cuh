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

	__host__ __device__ float x() const;

	__host__ __device__ float y() const;

	__host__ __device__ float z() const;

	__host__ __device__ Vector3D operator+(const float& t);

	__host__ __device__ Vector3D operator-(const float& t);

	__host__ __device__ Vector3D operator*(const float& t);

	__host__ __device__ Vector3D operator/(const float& t);

	__host__ __device__ Vector3D operator+(const Vector3D& vector);

	__host__ __device__ Vector3D operator-(const Vector3D& vector);

	__host__ __device__ Vector3D operator*(const Vector3D& vector);

	__host__ __device__ Vector3D operator/(const Vector3D& vector);

	__host__ __device__ Vector3D Squared();

	__host__ __device__ Vector3D Negative();

	__host__ __device__ float SquaredLength(const Vector3D& vector1);

	__host__ __device__ float Length(const Vector3D& vector1);

	__host__ __device__ Vector3D UnitVector(const Vector3D& vector);

	__host__ __device__ float DotProduct(const Vector3D& vector1, const Vector3D& vector2);

	__host__ __device__ Vector3D CrossProduct(const Vector3D& vector1, const Vector3D& vector2);

	__host__ __device__ Vector3D Truncate();

	__host__ __device__ Matrix ToMatrix();

	__host__ __device__ void Reset();

	__host__ __device__ Vector3D Clamp(float min, float max);

	float dim[3];
};