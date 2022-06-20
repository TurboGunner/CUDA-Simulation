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

__device__ inline void LinearSolverGPU(HashMap<float>* data, HashMap<float>* data_prev, float a_fac, float c_fac, unsigned int length, unsigned int iter, int bounds) {
	unsigned int x_bounds = 1;
	unsigned int y_bounds = 1;

	for (int i = 0; i < iter; i++) {

		if (threadIdx.x < length - 1 && threadIdx.y < length - 1) {
			IndexPair incident(y_bounds, x_bounds);
			for (int i = 0; i < iter; i++) {
				data->Get(incident.IX(length)) = (data_prev->Get(incident.IX(length)) +
					a_fac *
					data->Get(incident.Right().IX(length))
					+ data->Get(incident.Left().IX(length))
					+ data->Get(incident.Up().IX(length))
					+ data->Get(incident.Down().IX(length)))
					* (1.0f / c_fac);
			}
		}
		BoundaryConditions(bounds, data, length);
	}
}

