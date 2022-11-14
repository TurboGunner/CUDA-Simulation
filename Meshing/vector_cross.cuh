#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "matrix.cuh"

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
		dim[0] = vector.dim[0];
		dim[1] = vector.dim[1];
		dim[2] = vector.dim[2];
	}

	__host__ __device__ inline float x() const {
		return dim[0];
	}

	__host__ __device__ inline float y() const {
		return dim[1];
	}

	__host__ __device__ inline float z() const {
		return dim[2];
	}

	//Addition

	__host__ __device__ Vector3D operator+(const float t) {
		Vector3D output(dim[0], dim[1], dim[2]);

		output.dim[0] += t;
		output.dim[1] += t;
		output.dim[2] += t;

		return output;
	}

	__host__ __device__ Vector3D operator+(const Vector3D& vector) {
		Vector3D output(dim[0], dim[1], dim[2]);

		output.dim[0] += vector.dim[0];
		output.dim[1] += vector.dim[1];
		output.dim[2] += vector.dim[2];

		return output;
	}

	__host__ __device__ Vector3D& operator+=(const Vector3D& vector) {
		dim[0] += vector.dim[0];
		dim[1] += vector.dim[1];
		dim[2] += vector.dim[2];

		return *this;
	}

	__host__ __device__ Vector3D& operator+=(const float t) {
		dim[0] += t;
		dim[1] += t;
		dim[2] += t;

		return *this;
	}


	//Subtraction

	__host__ __device__ Vector3D operator-(const float t) {
		Vector3D output(dim[0], dim[1], dim[2]);

		output.dim[0] -= t;
		output.dim[1] -= t;
		output.dim[2] -= t;

		return output;
	}

	__host__ __device__ Vector3D operator-(const Vector3D& vector) {
		Vector3D output(dim[0], dim[1], dim[2]);

		output.dim[0] -= vector.dim[0];
		output.dim[1] -= vector.dim[1];
		output.dim[2] -= vector.dim[2];

		return output;
	}

	__host__ __device__ Vector3D& operator-=(const Vector3D& vector) {
		dim[0] -= vector.dim[0];
		dim[1] -= vector.dim[1];
		dim[2] -= vector.dim[2];

		return *this;
	}

	__host__ __device__ Vector3D& operator-=(const float t) {
		dim[0] -= t;
		dim[1] -= t;
		dim[2] -= t;

		return *this;
	}

	//Multiplication

	__host__ __device__ Vector3D operator*(const Vector3D& vector) {
		Vector3D output(dim[0], dim[1], dim[2]);

		output.dim[0] *= vector.dim[0];
		output.dim[1] *= vector.dim[1];
		output.dim[2] *= vector.dim[2];

		return output;
	}

	__host__ __device__ Vector3D operator*(const float t) {
		Vector3D output(dim[0], dim[1], dim[2]);

		output.dim[0] *= t;
		output.dim[1] *= t;
		output.dim[2] *= t;

		return output;
	}

	__host__ __device__ Vector3D& operator*=(const Vector3D& vector) {
		dim[0] *= vector.dim[0];
		dim[1] *= vector.dim[1];
		dim[2] *= vector.dim[2];

		return *this;
	}

	__host__ __device__ Vector3D& operator*=(const float t) {
		dim[0] *= t;
		dim[1] *= t;
		dim[2] *= t;

		return *this;
	}

	//Division

	__host__ __device__ Vector3D operator/(const Vector3D& vector) {
		Vector3D output(dim[0], dim[1], dim[2]);

		output.dim[0] /= vector.dim[0];
		output.dim[1] /= vector.dim[1];
		output.dim[2] /= vector.dim[2];

		return output;
	}

	__host__ __device__ Vector3D operator/(const float t) {
		Vector3D output(dim[0], dim[1], dim[2]);

		output.dim[0] /= t;
		output.dim[1] /= t;
		output.dim[2] /= t;

		return output;
	}

	__host__ __device__ Vector3D& operator/=(const Vector3D& vector) {
		dim[0] /= vector.dim[0];
		dim[1] /= vector.dim[1];
		dim[2] /= vector.dim[2];

		return *this;
	}

	__host__ __device__ Vector3D& operator/=(const float t) {
		dim[0] /= t;
		dim[1] /= t;
		dim[2] /= t;

		return *this;
	}

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
		return Vector3D(truncf(dim[0]), truncf(dim[1]), truncf(dim[2]));
	}

	__host__ __device__ Matrix<1, 3> ToMatrix() {
		Matrix<1, 3> matrix;
		for (size_t i = 0; i < 3; i++) {
			matrix.Set(dim[i], i);
		}
		return matrix;
	}

	__host__ __device__ void ToMatrix(Matrix<1, 3>* matrix) {
		for (size_t i = 0; i < 3; i++) {
			matrix->Set(dim[i], i);
		}
	}

	__host__ __device__ void Reset() {
		dim[0] = 0.0f;
		dim[1] = 0.0f;
		dim[2] = 0.0f;
	}

	__host__ __device__ Vector3D Clamp(const float min, const float max) {
		Vector3D output = {};

		for (size_t i = 0; i < 3; ++i) {
			const float temp = dim[i] < min ? min : dim[i];
			output.dim[i] = temp > max ? max : temp;
		}
		return output;
	}

	float dim[3];
};