#pragma once

#include "cuda_runtime.h"

#include "vector_field.hpp"
#include "fluid_sim.hpp"
#include "axis_data.hpp"

__device__ inline void LinearSolverGPU(HashMap<IndexPair, float, HashDupe<IndexPair>>* data, HashMap<IndexPair, float, HashDupe<IndexPair>>* data_prev, float a_fac, float c_fac, unsigned int length, unsigned int iter, int bounds) {
	unsigned int x_bounds = 1;
	unsigned int y_bounds = 1;

	for (int i = 0; i < iter; i++) {
		HashMap<Direction, IndexPair, HashFunc<Direction>>* pairs = GetAdjacentCoordinates(IndexPair(y_bounds, x_bounds));

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
		BoundaryConditions(bounds, data, length);
	}
}