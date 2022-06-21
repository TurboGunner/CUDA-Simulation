#pragma once

#include "cuda_runtime.h"

#include "vector_field.hpp"
#include "fluid_sim.hpp"
#include "axis_data.hpp"

__host__ __device__ inline void BoundaryConditions(int bounds, HashMap<F_Vector>* c_map, int length) {
	unsigned int bound = length - 1;

	for (int i = 1; i < bound; i++) {
		(*c_map)[IndexPair(i, 0).IX(length)] = bounds == 1 ? (*c_map)[IndexPair(i, 0).IX(length)] * -1.0f : (*c_map)[IndexPair(i, 1).IX(length)];
		(*c_map)[IndexPair(i, bound).IX(length)] = bounds == 1 ? (*c_map)[IndexPair(i, bound - 1).IX(length)] * -1.0f : (*c_map)[IndexPair(i, bound - 1).IX(length)];
	}
	for (int j = 1; j < bound; j++) {
		(*c_map)[IndexPair(0, j).IX(length)] = bounds == 1 ? (*c_map)[IndexPair(1, j).IX(length)] * -1.0f : (*c_map)[IndexPair(1, j).IX(length)];
		(*c_map)[IndexPair(bound, j).IX(length)] = bounds == 1 ? (*c_map)[IndexPair(bound - 1, j).IX(length)] * -1.0f : (*c_map)[IndexPair(bound - 1, j).IX(length)];
	}

	(*c_map)[IndexPair(0, 0).IX(length)] = (*c_map)[IndexPair(1, 0).IX(length)] + (*c_map)[IndexPair(0, 1).IX(length)] * .5f;
	(*c_map)[IndexPair(0, bound).IX(length)] = (*c_map)[IndexPair(1, bound).IX(length)] + (*c_map)[IndexPair(0, bound - 1).IX(length)] * .5f;
	(*c_map)[IndexPair(bound, 0).IX(length)] = (*c_map)[IndexPair(bound - 1, 0).IX(length)] + (*c_map)[IndexPair(bound, 1).IX(length)] * .5f;
	(*c_map)[IndexPair(bound, bound).IX(length)] = (*c_map)[IndexPair(bound - 1, bound).IX(length)] + (*c_map)[IndexPair(bound, bound - 1).IX(length)] * .5f;
}

__host__ __device__ inline void BoundaryConditions(int bounds, HashMap<float>* c_map, int length) {
	unsigned int bound = length - 1;

	for (int i = 1; i < bound; i++) {
		(*c_map)[IndexPair(i, 0).IX(length)] = bounds == 2 ? (*c_map)[IndexPair(i, 0).IX(length)] * -1.0f : (*c_map)[IndexPair(i, 1).IX(length)];
		(*c_map)[IndexPair(i, bound).IX(length)] = bounds == 2 ? (*c_map)[IndexPair(i, bound - 1).IX(length)] * -1.0f : (*c_map)[IndexPair(i, bound - 1).IX(length)];
	}
	for (int j = 1; j < bound; j++) {
		(*c_map)[IndexPair(0, j).IX(length)] = bounds == 1 ? (*c_map)[IndexPair(1, j).IX(length)] * -1.0f : (*c_map)[IndexPair(1, j).IX(length)];
		(*c_map)[IndexPair(bound, j).IX(length)] = bounds == 1 ? (*c_map)[IndexPair(bound - 1, j).IX(length)] * -1.0f : (*c_map)[IndexPair(bound - 1, j).IX(length)];
	}

	(*c_map)[IndexPair(0, 0).IX(length)] = (*c_map)[IndexPair(1, 0).IX(length)] + (*c_map)[IndexPair(0, 1).IX(length)] * .5f;
	(*c_map)[IndexPair(0, bound).IX(length)] = (*c_map)[IndexPair(1, bound).IX(length)] + (*c_map)[IndexPair(0, bound - 1).IX(length)] * .5f;
	(*c_map)[IndexPair(bound, 0).IX(length)] = (*c_map)[IndexPair(bound - 1, 0).IX(length)] + (*c_map)[IndexPair(bound, 1).IX(length)] * .5f;
	(*c_map)[IndexPair(bound, bound).IX(length)] = (*c_map)[IndexPair(bound - 1, bound).IX(length)] + (*c_map)[IndexPair(bound, bound - 1).IX(length)] * .5f;
}