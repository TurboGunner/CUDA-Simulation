#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <math.h>

class Vector3D {
public:
	__host__ __device__ Vector3D() = default;

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

	__host__ __device__ Vector3D AddByScalar(const Vector3D& multiplier, const float& t) {
		Vector3D output = multiplier;

		output.dim[0] += t;
		output.dim[1] += t;
		output.dim[2] += t;

		return output;
	}

	__host__ __device__ Vector3D SubtractByScalar(const Vector3D& multiplier, const float& t) {
		Vector3D output = multiplier;

		output.dim[0] -= t;
		output.dim[1] -= t;
		output.dim[2] -= t;

		return output;
	}

	__host__ __device__ Vector3D MultiplyByScalar(const Vector3D& multiplier, const float& t) {
		Vector3D output = multiplier;

		output.dim[0] *= t;
		output.dim[1] *= t;
		output.dim[2] *= t;

		return output;
	}

	__host__ __device__ Vector3D DivideByScalar(const Vector3D& multiplier, const float& t) {
		Vector3D output = multiplier;

		output.dim[0] /= t;
		output.dim[1] /= t;
		output.dim[2] /= t;

		return output;
	}

	__host__ __device__ Vector3D operator+(const Vector3D& vector) {

		dim[0] += vector.dim[0];
		dim[1] += vector.dim[1];
		dim[2] += vector.dim[2];

		return *this;
	}

	__host__ __device__ Vector3D SubtractVector(const Vector3D& vector1, const Vector3D& vector2) {
		Vector3D output = vector1;

		output.dim[0] -= vector2.dim[0];
		output.dim[1] -= vector2.dim[1];
		output.dim[2] -= vector2.dim[2];

		return output;
	}


	__host__ __device__ Vector3D MultiplyVector(const Vector3D& vector1, const Vector3D& vector2) {
		Vector3D output = vector1;

		output.dim[0] *= vector2.dim[0];
		output.dim[1] *= vector2.dim[1];
		output.dim[2] *= vector2.dim[2];

		return output;
	}

	__host__ __device__ float SquaredLength(const Vector3D& vector1) {
		return vector1.dim[0] * vector1.dim[0] + vector1.dim[1] * vector1.dim[1] + vector1.dim[2] * vector1.dim[2];
	}

	__host__ __device__ float Length(const Vector3D& vector1) {
		return sqrt(vector1.dim[0] * vector1.dim[0] + vector1.dim[1] * vector1.dim[1] + vector1.dim[2] * vector1.dim[2]);
	}

	__host__ __device__ inline Vector3D UnitVector(const Vector3D& vector) {
		Vector3D output = vector;
		output = DivideByScalar(vector, Length(vector));

		return output;
	}

	__host__ __device__ float DotProduct(const Vector3D& vector1, const Vector3D& vector2) {
		return (vector1.dim[0] * vector2.dim[0]) + (vector1.dim[1] * vector2.dim[1]) + (vector1.dim[2] * vector2.dim[2]);
	}

	__host__ __device__ Vector3D CrossProduct(const Vector3D& vector1, const Vector3D& vector2) {
		return Vector3D((vector1.dim[1] * vector2.dim[2] - vector1.dim[2] * vector2.dim[1]),
			(-(vector1.dim[0] * vector2.dim[2] - vector1.dim[2] * vector2.dim[0])),
			(vector1.dim[0] * vector2.dim[1] - vector1.dim[1] * vector2.dim[0]));
	}

	float dim[3];


};