#pragma once

#include "cuda_runtime.h"

#include "vector_field.hpp"
#include "fluid_sim.hpp"
#include "axis_data.hpp"

__host__ __device__ inline void BoundaryConditions(int bounds, HashMap<IndexPair, F_Vector, Hash<IndexPair>>* c_map, int side_size) {
	unsigned int bound = side_size - 1;

	for (int i = 1; i < bound; i++) {
		(*c_map)[IndexPair(i, 0)].vx_ = bounds == 1 ? (*c_map)[IndexPair(i, 1)].vx_ * -1.0f : (*c_map)[IndexPair(i, 1)].vx_;
		(*c_map)[IndexPair(i, bound)].vx_ = bounds == 1 ? (*c_map)[IndexPair(i, bound - 1)].vx_ * -1.0f : (*c_map)[IndexPair(i, bound - 1)].vx_;
	}
	for (int j = 1; j < bound; j++) {
		(*c_map)[IndexPair(0, j)].vy_ = bounds == 1 ? (*c_map)[IndexPair(1, j)].vy_ * -1.0f : (*c_map)[IndexPair(1, j)].vy_;
		(*c_map)[IndexPair(bound, j)].vy_ = bounds == 1 ? (*c_map)[IndexPair(bound - 1, j)].vy_ * -1.0f : (*c_map)[IndexPair(bound - 1, j)].vy_;
	}

	(*c_map)[IndexPair(0, 0)] = (*c_map)[IndexPair(1, 0)] + (*c_map)[IndexPair(0, 1)] * .5f;
	(*c_map)[IndexPair(0, bound)] = (*c_map)[IndexPair(1, bound)] + (*c_map)[IndexPair(0, bound - 1)] * .5f;
	(*c_map)[IndexPair(bound, 0)] = (*c_map)[IndexPair(bound - 1, 0)] + (*c_map)[IndexPair(bound, 1)] * .5f;
	(*c_map)[IndexPair(bound, bound)] = (*c_map)[IndexPair(bound - 1, bound)] + (*c_map)[IndexPair(bound, bound - 1)] * .5f;
}

__host__ __device__ inline void BoundaryConditions(int bounds, HashMap<IndexPair, float, HashDupe<IndexPair>>* c_map, int side_size) {
	unsigned int bound = side_size - 1;

	for (int i = 1; i < bound; i++) {
		(*c_map)[IndexPair(i, 0)] = bounds == 2 ? (*c_map)[IndexPair(i, 0)] * -1.0f : (*c_map)[IndexPair(i, 1)];
		(*c_map)[IndexPair(i, bound)] = bounds == 2 ? (*c_map)[IndexPair(i, bound - 1)] * -1.0f : (*c_map)[IndexPair(i, bound - 1)];
	}
	for (int j = 1; j < bound; j++) {
		(*c_map)[IndexPair(0, j)] = bounds == 1 ? (*c_map)[IndexPair(1, j)] * -1.0f : (*c_map)[IndexPair(1, j)];
		(*c_map)[IndexPair(bound, j)] = bounds == 1 ? (*c_map)[IndexPair(bound - 1, j)] * -1.0f : (*c_map)[IndexPair(bound - 1, j)];
	}

	(*c_map)[IndexPair(0, 0)] = (*c_map)[IndexPair(1, 0)] + (*c_map)[IndexPair(0, 1)] * .5f;
	(*c_map)[IndexPair(0, bound)] = (*c_map)[IndexPair(1, bound)] + (*c_map)[IndexPair(0, bound - 1)] * .5f;
	(*c_map)[IndexPair(bound, 0)] = (*c_map)[IndexPair(bound - 1, 0)] + (*c_map)[IndexPair(bound, 1)] * .5f;
	(*c_map)[IndexPair(bound, bound)] = (*c_map)[IndexPair(bound - 1, bound)] + (*c_map)[IndexPair(bound, bound - 1)] * .5f;
}

__device__ inline void LinearSolverGPU(HashMap<IndexPair, float, HashDupe<IndexPair>>* data, HashMap<IndexPair, float, HashDupe<IndexPair>>* data_prev, float a_fac, float c_fac, unsigned int length, unsigned int iter, int bounds) {
	unsigned int x_bounds = 1;
	unsigned int y_bounds = 1;

	for (int i = 0; i < iter; i++) {

		if (threadIdx.x < length - 1 && threadIdx.y < length - 1) {
			IndexPair incident(y_bounds, x_bounds);
			for (int i = 0; i < iter; i++) {
				(*data)[incident] = ((*data_prev)[incident] +
					a_fac *
					(*data)[incident.Right()]
					+ (*data)[incident.Left()]
					+ (*data)[incident.Up()]
					+ (*data)[incident.Down()])
					* (1.0f / c_fac);
			}
		}
		BoundaryConditions(bounds, data, length);
	}
}

