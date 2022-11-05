#include "vector_cross.cuh"

__host__ __device__ Vector3D::Vector3D(float x_in, float y_in, float z_in) {
	dim[0] = x_in;
	dim[1] = y_in;
	dim[2] = z_in;
}

__host__ __device__  void Vector3D::operator=(const Vector3D& vector) {
	dim[0] = vector.dim[0];
	dim[1] = vector.dim[1];
	dim[2] = vector.dim[2];
}

__host__ __device__ Vector3D Vector3D::operator+(const float t) {
	Vector3D output(dim[0], dim[1], dim[2]);

	output.dim[0] += t;
	output.dim[1] += t;
	output.dim[2] += t;

	return output;
}

__host__ __device__ Vector3D Vector3D::operator-(const float t) {
	Vector3D output(dim[0], dim[1], dim[2]);

	output.dim[0] -= t;
	output.dim[1] -= t;
	output.dim[2] -= t;

	return output;
}

__host__ __device__ Vector3D Vector3D::operator*(const float t) {
	Vector3D output(dim[0], dim[1], dim[2]);

	output.dim[0] *= t;
	output.dim[1] *= t;
	output.dim[2] *= t;

	return output;
}

__host__ __device__ Vector3D Vector3D::operator/(const float t) {
	Vector3D output(dim[0], dim[1], dim[2]);

	output.dim[0] /= t;
	output.dim[1] /= t;
	output.dim[2] /= t;

	return output;
}

__host__ __device__ Vector3D Vector3D::operator+(const Vector3D& vector) {
	Vector3D output(dim[0], dim[1], dim[2]);

	output.dim[0] += vector.dim[0];
	output.dim[1] += vector.dim[1];
	output.dim[2] += vector.dim[2];

	return output;
}

__host__ __device__ Vector3D Vector3D::operator-(const Vector3D& vector) {
	Vector3D output(dim[0], dim[1], dim[2]);

	output.dim[0] -= vector.dim[0];
	output.dim[1] -= vector.dim[1];
	output.dim[2] -= vector.dim[2];

	return output;
}

__host__ __device__ Vector3D Vector3D::operator*(const Vector3D& vector) {
	Vector3D output(dim[0], dim[1], dim[2]);

	output.dim[0] *= vector.dim[0];
	output.dim[1] *= vector.dim[1];
	output.dim[2] *= vector.dim[2];

	return output;
}

__host__ __device__ Vector3D& Vector3D::operator*=(const float t) {
	dim[0] *= t;
	dim[1] *= t;
	dim[2] *= t;

	return *this;
}

__host__ __device__ Vector3D Vector3D::operator/(const Vector3D& vector) {
	Vector3D output(dim[0], dim[1], dim[2]);

	output.dim[0] /= vector.dim[0];
	output.dim[1] /= vector.dim[1];
	output.dim[2] /= vector.dim[2];

	return output;
}

__host__ __device__ Vector3D& Vector3D::operator+=(const Vector3D& vector) {
	dim[0] += vector.dim[0];
	dim[1] += vector.dim[1];
	dim[2] += vector.dim[2];

	return *this;
}

__host__ __device__ Vector3D& Vector3D::operator/=(const Vector3D& vector) {
	dim[0] /= vector.dim[0];
	dim[1] /= vector.dim[1];
	dim[2] /= vector.dim[2];

	return *this;
}

__host__ __device__ Vector3D& Vector3D::operator/=(const float t) {
	dim[0] /= t;
	dim[1] /= t;
	dim[2] /= t;

	return *this;
}

__host__ __device__ Matrix Vector3D::ToMatrix() {
	Matrix matrix(1, 3);
	for (int i = 0; i < 3; i++) {
		matrix.Get(i) = dim[i];
	}
	return matrix;
}

__host__ __device__ void Vector3D::ToMatrix(Matrix* matrix) {
	assert(matrix->rows * matrix->columns == 3);
	for (int i = 0; i < matrix->rows * matrix->columns; i++) {
		matrix->Get(i) = dim[i];
	}
}

__host__ __device__ void Vector3D::Reset() {
	dim[0] = 0.0f;
	dim[1] = 0.0f;
	dim[2] = 0.0f;
}

__host__ __device__ Vector3D Vector3D::Clamp(const float min, const float max) {
	Vector3D output(dim[0], dim[1], dim[2]);

	for (size_t i = 0; i < 3; i++) {
		dim[i] = dim[i] < min ? min : dim[i];
		dim[i] = dim[i] > max ? max : dim[i];
	}

	return output;
}