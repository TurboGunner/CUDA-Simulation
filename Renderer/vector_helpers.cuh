#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

class Vector3D {
public:
	__host__ __device__ Vector3D() {};

	__host__ __device__ Vector3D(float x_in, float y_in, float z_in) {
		dim[0] = x_in;
		dim[1] = y_in;
		dim[2] = z_in;
	}

	__host__ __device__  void operator=(const Vector3D& vector) {
		dim[0] = vector.x();
		dim[1] = vector.y();
		dim[2] = vector.z();
	}

	__host__ __device__ float x() const {
		return dim[0];
	}

	__host__ __device__ float y() const {
		return dim[1];
	}

	__host__ __device__ float z() const {
		return dim[2];
	}

	float dim[3];
};

__host__ __device__ inline Vector3D AddByScalar(const Vector3D& multiplier, const float& t) {
	Vector3D output = multiplier;

	output.dim[0] += t;
	output.dim[1] += t;
	output.dim[2] += t;

	return output;
}

__host__ __device__ inline Vector3D SubtractByScalar(const Vector3D& multiplier, const float& t) {
	Vector3D output = multiplier;

	output.dim[0] -= t;
	output.dim[1] -= t;
	output.dim[2] -= t;

	return output;
}

__host__ __device__ inline Vector3D MultiplyByScalar(const Vector3D& multiplier, const float& t) {
	Vector3D output = multiplier;

	output.dim[0] *= t;
	output.dim[1] *= t;
	output.dim[2] *= t;

	return output;
}

__host__ __device__ inline Vector3D DivideByScalar(const Vector3D& multiplier, const float& t) {
	Vector3D output = multiplier;

	output.dim[0] /= t;
	output.dim[1] /= t;
	output.dim[2] /= t;

	return output;
}

__host__ __device__ inline Vector3D AddVector(const Vector3D& vector1, const Vector3D& vector2) {
	Vector3D output = vector1;

	output.dim[0] += vector2.dim[0];
	output.dim[1] += vector2.dim[1];
	output.dim[2] += vector2.dim[2];

	return output;
}

__host__ __device__ inline Vector3D SubtractVector(const Vector3D& vector1, const Vector3D& vector2) {
	Vector3D output = vector1;

	output.dim[0] -= vector2.dim[0];
	output.dim[1] -= vector2.dim[1];
	output.dim[2] -= vector2.dim[2];

	return output;
}


__host__ __device__ inline Vector3D MultiplyVector(const Vector3D& vector1, const Vector3D& vector2) {
	Vector3D output = vector1;

	output.dim[0] *= vector2.dim[0];
	output.dim[1] *= vector2.dim[1];
	output.dim[2] *= vector2.dim[2];

	return output;
}

__host__ __device__ inline Vector3D UnitVector(const Vector3D& vector) {
	Vector3D output;
	float length = vector.x() + vector.y() + vector.z();

	output.dim[0] = vector.x() / length;
	output.dim[1] = vector.y() / length;
	output.dim[2] = vector.z() / length;

	return output;
}

__host__ __device__ inline float DotProduct(const Vector3D& vector1, const Vector3D& vector2) {
	return (vector1.dim[0] * vector2.dim[0]) + (vector1.dim[1] * vector2.dim[1]) + (vector1.dim[2] * vector2.dim[2]);
}

__host__ __device__ inline Vector3D CrossProduct(const Vector3D& vector1, const Vector3D& vector2) {
	return Vector3D((vector1.dim[1] * vector2.dim[2] - vector1.dim[2] * vector2.dim[1]),
		(-(vector1.dim[0] * vector2.dim[2] - vector1.dim[2] * vector2.dim[0])),
		(vector1.dim[0] * vector2.dim[1] - vector1.dim[1] * vector2.dim[0]));
}

__host__ __device__ inline float SquaredLength(const Vector3D& vector1) {
	return vector1.dim[0] * vector1.dim[0] + vector1.dim[1] * vector1.dim[1] + vector1.dim[2] * vector1.dim[2];
}

__host__ __device__ inline float Length(const Vector3D& vector1) {
	return sqrt(vector1.dim[0] * vector1.dim[0] + vector1.dim[1] * vector1.dim[1] + vector1.dim[2] * vector1.dim[2]);
}