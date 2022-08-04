#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

class Vector3D {
public:
	__host__ __device__ Vector3D() {};

	__host__ __device__ Vector3D(float x_in, float y_in, float z_in) {
		x = x_in;
		y = y_in;
		z = z_in;
	}

	__host__ __device__  void operator=(const Vector3D& vector) {
		x = vector.x;
		y = vector.y;
		z = vector.z;
	}

	float x = 0;
	float y = 0;
	float z = 0;
};

__device__ inline Vector3D AddByScalar(const Vector3D& multiplier, const float& t) {
	Vector3D output = multiplier;

	output.x += t;
	output.y += t;
	output.z += t;

	return output;
}

__device__ inline Vector3D SubtractByScalar(const Vector3D& multiplier, const float& t) {
	Vector3D output = multiplier;

	output.x -= t;
	output.y -= t;
	output.z -= t;

	return output;
}

__device__ inline Vector3D MultiplyByScalar(const Vector3D& multiplier, const float& t) {
	Vector3D output = multiplier;

	output.x *= t;
	output.y *= t;
	output.z *= t;

	return output;
}

__device__ inline Vector3D DivideByScalar(const Vector3D& multiplier, const float& t) {
	Vector3D output = multiplier;

	output.x /= t;
	output.y /= t;
	output.z /= t;

	return output;
}

__device__ inline Vector3D AddVector(const Vector3D& vector1, const Vector3D& vector2) {
	Vector3D output = vector1;

	output.x += vector2.x;
	output.y += vector2.y;
	output.z += vector2.z;

	return output;
}

__device__ inline Vector3D SubtractVector(const Vector3D& vector1, const Vector3D& vector2) {
	Vector3D output = vector1;

	output.x -= vector2.x;
	output.y -= vector2.y;
	output.z -= vector2.z;

	return output;
}


__device__ inline Vector3D MultiplyVector(const Vector3D& vector1, const Vector3D& vector2) {
	Vector3D output = vector1;

	output.x *= vector2.x;
	output.y *= vector2.y;
	output.z *= vector2.z;

	return output;
}

__device__ inline Vector3D UnitVector(const Vector3D& vector) {
	Vector3D output;
	float length = vector.x + vector.y + vector.z;

	output.x = vector.x / length;
	output.y = vector.y / length;
	output.z = vector.z / length;

	return output;
}

__device__ inline float DotProduct(const Vector3D& vector1, const Vector3D& vector2) {
	return (vector1.x * vector2.x) + (vector1.y * vector2.y) + (vector1.z * vector2.z);
}

__device__ inline Vector3D CrossProduct(const Vector3D& vector1, const Vector3D& vector2) {
	return Vector3D((vector1.y * vector2.z - vector1.z * vector2.y),
		(-(vector1.x * vector2.z - vector1.z * vector2.x)),
		(vector1.x * vector2.y - vector1.y * vector2.x));
}

__device__ inline float SquaredLength(const Vector3D& vector1) {
	return vector1.x * vector1.x + vector1.y * vector1.y + vector1.z * vector1.z;
}

__device__ inline float Length(const Vector3D& vector1) {
	return sqrt(vector1.x * vector1.x + vector1.y * vector1.y + vector1.z * vector1.z);
}