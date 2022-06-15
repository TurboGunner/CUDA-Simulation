#pragma once

#include "cuda_runtime.h"

#include "vector_field.hpp"
#include "fluid_sim.hpp"
#include "axis_data.hpp"

__device__ inline int IX(unsigned int x, unsigned int y, const unsigned int& size) {
	unsigned int value = (((y - 1) * size) + x);
	if (value < (size * size)) {
		return value;
	}
}

__device__ inline void PointerBoundaries(HashMap<IndexPair, float, HashDupe<IndexPair>>* result_ptr, const unsigned int& length) {
	unsigned int bound = length - 1;
	(*result_ptr)[IX(0, 1, length)] = (*result_ptr)[IX(1, 1, length)] + (*result_ptr)[IX(0, 2, length)] * .5f;
	(*result_ptr)[IX(0, length, length)] = (*result_ptr)[IX(1, length, length)] + (*result_ptr)[IX(0, bound, length)] * .5f;
	(*result_ptr)[IX(bound, 1, length)] = (*result_ptr)[IX(bound - 1, 1, length)] + (*result_ptr)[IX(bound, 2, length)] * .5f;
	(*result_ptr)[IX(bound, length, length)] = (*result_ptr)[IX(bound - 1, length, length)] + (*result_ptr)[IX(bound, bound, length)] * .5f;
}

__device__ inline void PointerBoundariesSpecialX(HashMap<IndexPair, float, HashDupe<IndexPair>>* result_ptr, const unsigned int& length) {
	unsigned int bound = length - 1;
	for (int i = 1; i < bound; i++) {
		(*result_ptr)[IX(i, 1, length)] = (*result_ptr)[IX(i, 2, length)] * -1;
		(*result_ptr)[IX(i, length, length)] = (*result_ptr)[IX(i, bound, length)] * -1;
	}
	PointerBoundaries(result_ptr, length);
}

__device__ inline void PointerBoundariesSpecialY(HashMap<IndexPair, float, HashDupe<IndexPair>>* result_ptr, const unsigned int& length) {
	unsigned int bound = length - 1;
	for (int j = 1; j < bound; j++) {
		(*result_ptr)[IX(0, j + 1, length)] = -(*result_ptr)[IX(1, j + 1, length)];
		(*result_ptr)[IX(bound, j + 1, length)] = -(*result_ptr)[IX(bound - 1, j + 1, length)];
	}
	PointerBoundaries(result_ptr, length);
}

__device__ inline void LinearSolverGPU(HashMap<IndexPair, float, HashDupe<IndexPair>>* data, HashMap<IndexPair, float, HashDupe<IndexPair>>* data_prev, float a_fac, float c_fac, unsigned int length, unsigned int iter, int bounds) {
	unsigned int x_bounds = 1;
	unsigned int y_bounds = 1;

	for (int i = 0; i < iter; i++) {
		HashMap<Direction, IndexPair, HashFunc<Direction>>* pairs = GetAdjacentCoordinates(IndexPair(y_bounds, x_bounds), data->size_);

		if (threadIdx.x < length - 1 && threadIdx.y < length - 1) {
			for (int i = 0; i < iter; i++) {
				(*data)[pairs->Get(Direction::Origin)] = ((*data_prev)[pairs->Get(Direction::Origin)] +
					a_fac *
					(*data)[pairs->Get(Direction::Right)]
					+ (*data)[pairs->Get(Direction::Left)]
					+ (*data)[pairs->Get(Direction::Up)]
					+ (*data)[pairs->Get(Direction::Down)])
					* (1.0f / c_fac);
			}
		}
		if (bounds == 0) {
			PointerBoundaries(data, length);
		}
		if (bounds == 1) {
			PointerBoundariesSpecialX(data, length);
		}
		else {
			PointerBoundariesSpecialY(data, length);
		}
	}
}