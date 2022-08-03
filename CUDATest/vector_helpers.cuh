#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

__device__ inline float3 AddByScalar(const float3& multiplier, const float& t) {
	float3 output = multiplier;

	output.x += t;
	output.y += t;
	output.z += t;

	return output;
}

__device__ inline float3 SubtractByScalar(const float3& multiplier, const float& t) {
	float3 output = multiplier;

	output.x -= t;
	output.y -= t;
	output.z -= t;

	return output;
}

__device__ inline float3 MultiplyByScalar(const float3& multiplier, const float& t) {
	float3 output = multiplier;

	output.x *= t;
	output.y *= t;
	output.z *= t;

	return output;
}

__device__ inline float3 DivideByScalar(const float3& multiplier, const float& t) {
	float3 output = multiplier;

	output.x /= t;
	output.y /= t;
	output.z /= t;

	return output;
}

__device__ inline float3 AddVector(const float3& vector1, const float3& vector2) {
	float3 output = vector1;

	output.x += vector2.x;
	output.y += vector2.y;
	output.z += vector2.z;

	return output;
}

__device__ inline float3 SubtractVector(const float3& vector1, const float3& vector2) {
	float3 output = vector1;

	output.x -= vector2.x;
	output.y -= vector2.y;
	output.z -= vector2.z;

	return output;
}


__device__ inline float3 MultiplyVector(const float3& vector1, const float3& vector2) {
	float3 output = vector1;

	output.x *= vector2.x;
	output.y *= vector2.y;
	output.z *= vector2.z;

	return output;
}

__device__ inline float3 UnitVector(const float3& vector) {
	float3 output;
	float length = vector.x + vector.y + vector.z;

	output.x = vector.x / length;
	output.y = vector.y / length;
	output.z = vector.z / length;

	return output;
}

__device__ inline float DotProduct(const float3& vector1, const float3& vector2) {
	return (vector1.x * vector2.x) + (vector1.y * vector2.y) + (vector1.z * vector2.z);
}

__device__ inline float3 CrossProduct(const float3& vector1, const float3& vector2) {
	return float3((vector1.y * vector2.z - vector1.z * vector2.y),
		(-(vector1.x * vector2.z - vector1.z * vector2.x)),
		(vector1.x * vector2.y - vector1.y * vector2.x));
}

__device__ inline float SquaredLength(const float3& vector1) {
	return vector1.x * vector1.x + vector1.y * vector1.y + vector1.z * vector1.z;
}