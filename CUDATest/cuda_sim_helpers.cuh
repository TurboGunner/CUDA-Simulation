#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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