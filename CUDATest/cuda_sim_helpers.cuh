#pragma once

#include "cuda_runtime.h"

__device__ inline int IX(unsigned int x, unsigned int y, const unsigned int& size) {
	unsigned int value = (((y - 1) * size) + x);
	if (value < (size * size)) {
		return value;
	}
}

__device__ inline void PointerBoundaries(float* result_ptr, const unsigned int& length) {
	unsigned int bound = length - 1;
	result_ptr[IX(0, 1, length)] = result_ptr[IX(1, 1, length)] + result_ptr[IX(0, 2, length)] * .5f;
	result_ptr[IX(0, length, length)] = result_ptr[IX(1, length, length)] + result_ptr[IX(0, bound, length)] * .5f;
	result_ptr[IX(bound, 1, length)] = result_ptr[IX(bound - 1, 1, length)] + result_ptr[IX(bound, 2, length)] * .5f;
	result_ptr[IX(bound, length, length)] = result_ptr[IX(bound - 1, length, length)] + result_ptr[IX(bound, bound, length)] * .5f;
}

__device__ inline void PointerBoundariesSpecial(float* result_ptr, const unsigned int& length) {
	unsigned int bound = length - 1;
	for (int i = 1; i < bound; i++) {
		result_ptr[IX(i, 1, length)] *= -1.0f;
		result_ptr[IX(i, length, length)] *= -1.0f;
	}
	for (int j = 1; j < bound; j++) {
		result_ptr[IX(0, j + 1, length)] *= -1.0f;
		result_ptr[IX(bound, j + 1, length)] *= -1.0f;
	}
	PointerBoundaries(result_ptr, length);
}

__device__ inline void PointerBoundariesSpecial(float3* vector_ptr, const unsigned int& length) {
	float* x_dim = new float[length], * y_dim = new float[length], * z_dim = new float[length];

	for (int i = 0; i < length * length; i++) {
		x_dim[i] = vector_ptr[i].x;
	}

	float* result_ptr = x_dim;

	for (int x = 0; x <= 1; x++) { //Only 1 for now due to two dimensions
		if (x == 1) {
			result_ptr = y_dim;
		}
		else if (x == 2) {
			result_ptr = z_dim;
		}
		PointerBoundariesSpecial(result_ptr, length);
	}
}

__device__ inline void LinearSolverGPU(float* data, const float* data_prev, float a_fac, float c_fac, unsigned int length, unsigned int iter) {
	unsigned int x_bounds = 1;
	unsigned int y_bounds = 1;

	for (int i = 0; i < iter; i++) {
		for (y_bounds; y_bounds < length - 1; y_bounds++) {
			for (x_bounds; x_bounds < length - 1; x_bounds++) {
				data[IX(x_bounds, y_bounds + 1, length)] =
					(data_prev[IX(x_bounds, y_bounds + 1, length)] +
						a_fac *
						(data[IX(x_bounds + 1, y_bounds + 1, length)]
						+ data[IX(x_bounds - 1, y_bounds + 1, length)]
						+ data[IX(x_bounds, y_bounds + 2, length)]
						+ data[IX(x_bounds, y_bounds, length)]))
					* (1.0f / c_fac);
			}
		}
		PointerBoundaries(data, length);
	}
}