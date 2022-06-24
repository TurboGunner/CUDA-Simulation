#include "fluid_sim_cuda.cuh"

__host__ __device__ void BoundaryConditions(int bounds, HashMap<float>* c_map, int length) {
	unsigned int bound = length - 1;

	for (int i = 1; i < bound; i++) {
		for (int j = 1; j < bound; j++) {
			(*c_map)[IndexPair(i, j, 0).IX(length)] = bounds == 3 ? -(*c_map)[IndexPair(i, j, 1).IX(length)] : (*c_map)[IndexPair(i, j, 1).IX(length)];
			(*c_map)[IndexPair(i, j, bound).IX(length)] = bounds == 3 ? -(*c_map)[IndexPair(i, j, bound - 1).IX(length)] : (*c_map)[IndexPair(i, j, bound - 1).IX(length)];
		}
	}

	for (int i = 1; i < bound; i++) {
		for (int k = 1; k < bound; k++) {
			(*c_map)[IndexPair(i, 0, k).IX(length)] = bounds == 2 ? -(*c_map)[IndexPair(i, 1, k).IX(length)] : (*c_map)[IndexPair(i, 1, k).IX(length)];
			(*c_map)[IndexPair(i, bound, k).IX(length)] = bounds == 2 ? -(*c_map)[IndexPair(i, bound - 1, k).IX(length)] : (*c_map)[IndexPair(i, bound - 1, k).IX(length)];
		}
	}
	for (int j = 1; j < bound; j++) {
		for (int k = 1; k < bound; k++) {
			(*c_map)[IndexPair(0, j, k).IX(length)] = bounds == 1 ? -(*c_map)[IndexPair(1, j, k).IX(length), k] : (*c_map)[IndexPair(1, j, k).IX(length)];
			(*c_map)[IndexPair(bound, j, k).IX(length)] = bounds == 1 ? -(*c_map)[IndexPair(bound - 1, j, k).IX(length)] : (*c_map)[IndexPair(bound - 1, j, k).IX(length)];
		}
	}


	(*c_map)[IndexPair(0, 0, 0).IX(length)] = .33f * //Min X, Min Y, Min Z
		((*c_map)[IndexPair(1, 0, 0).IX(length)]
			+ (*c_map)[IndexPair(0, 1, 0).IX(length)]
			+ (*c_map)[IndexPair(0, 0, 1).IX(length)]);

	(*c_map)[IndexPair(0, bound, 0).IX(length)] = .33f * //Min X, Max Y, Min Z
		((*c_map)[IndexPair(1, bound, 0).IX(length)]
			+ (*c_map)[IndexPair(0, bound - 1, 0).IX(length)]
			+ (*c_map)[IndexPair(0, bound, 1).IX(length)]);

	(*c_map)[IndexPair(0, 0, bound).IX(length)] = .33f * //Min X, Min Y, Max Z
		((*c_map)[IndexPair(1, 0, bound).IX(length)]
			+ (*c_map)[IndexPair(0, 1, bound).IX(length)]
			+ (*c_map)[IndexPair(0, 0, length).IX(length)]);

	(*c_map)[IndexPair(bound, 0, 0).IX(length)] = .33f * //Max X, Min Y, Min Z
		((*c_map)[IndexPair(bound - 1, 0, 0).IX(length)]
			+ (*c_map)[IndexPair(bound, 1, 0).IX(length)]
			+ (*c_map)[IndexPair(bound, 0, 1).IX(length)]);

	(*c_map)[IndexPair(bound, bound, 0).IX(length)] = .33f * //Max X, Max Y, Min Z
		((*c_map)[IndexPair(bound - 1, bound, 0).IX(length)]
			+ (*c_map)[IndexPair(bound, bound - 1, 0).IX(length)]
			+ (*c_map)[IndexPair(bound, bound, 1).IX(length)]);

	(*c_map)[IndexPair(bound, 0, bound).IX(length)] = .33f * //Max X, Min Y, Max Z
		((*c_map)[IndexPair(bound - 1, 0, bound).IX(length)]
			+ (*c_map)[IndexPair(bound, 1, bound).IX(length)]
			+ (*c_map)[IndexPair(bound, 0, bound - 1).IX(length)]);


	(*c_map)[IndexPair(0, bound, bound).IX(length)] = .33f * //Max X, Min Y, Max Z
		((*c_map)[IndexPair(1, bound, bound).IX(length)]
			+ (*c_map)[IndexPair(0, bound - 1, bound).IX(length)]
			+ (*c_map)[IndexPair(0, bound, bound - 1).IX(length)]);


	(*c_map)[IndexPair(bound, bound, bound).IX(length)] = .33f * //Max X, Max Y, Max Z
		((*c_map)[IndexPair(bound - 1, bound, bound).IX(length)]
			+ (*c_map)[IndexPair(bound, bound - 1, bound).IX(length)]
			+ (*c_map)[IndexPair(bound, bound, bound - 1).IX(length)]);
}