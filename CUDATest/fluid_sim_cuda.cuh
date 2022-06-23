#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudamap.cuh"
#include "index_pair_cuda.cuh"

#include "vector_field.hpp"
#include "handler_methods.hpp"
#include "fluid_sim.hpp"
#include "axis_data.hpp"

#include <iostream>
#include <functional>
#include <vector>

using std::reference_wrapper;
using std::vector;
using std::function;

__global__ void LinearSolverKernel(HashMap<float>* data, HashMap<float>* data_prev, float a_fac, float c_fac, unsigned int length, unsigned int iter, int bounds);

void LinearSolverCuda(int bounds, AxisData& current, AxisData& previous, const float& a_fac, const float& c_fac, const unsigned int& iter, const unsigned int& length);

__global__ void AdvectKernel(HashMap<float>* data, HashMap<float>* data_prev, HashMap<float>* velocity_x, HashMap<float>* velocity_y, float dt, unsigned int length, int bounds);

void AdvectCuda(int bounds, AxisData& current, AxisData& previous, VectorField& velocity, const float& dt, const unsigned int& length);

__global__ void ProjectKernel(HashMap<float>* velocity_x, HashMap<float>* velocity_y, HashMap<float>* data, HashMap<float>* data_prev, unsigned int length);

__global__ void ProjectKernel2(HashMap<float>* velocity_x, HashMap<float>* velocity_y, HashMap<float>* data, HashMap<float>* data_prev, unsigned int length);

void ProjectCuda(int bounds, VectorField& velocity, VectorField& velocity_prev, const unsigned int& length, const unsigned int& iter);

__host__ __device__ inline void BoundaryConditions(int bounds, HashMap<float>* c_map, int length) {
	unsigned int bound = length - 1;
	for (int i = 1; i < bound; i++) {
		(*c_map)[IndexPair(i, 0).IX(length)] = bounds == 2 ? -(*c_map)[IndexPair(i, 1).IX(length)] : (*c_map)[IndexPair(i, 1).IX(length)];
		(*c_map)[IndexPair(i, bound).IX(length)] = bounds == 2 ? -(*c_map)[IndexPair(i, bound - 1).IX(length)] : (*c_map)[IndexPair(i, bound - 1).IX(length)];
	}
	for (int j = 1; j < bound; j++) {
		(*c_map)[IndexPair(0, j).IX(length)] = bounds == 1 ? -(*c_map)[IndexPair(1, j).IX(length)] : (*c_map)[IndexPair(1, j).IX(length)];
		(*c_map)[IndexPair(bound, j).IX(length)] = bounds == 1 ? -(*c_map)[IndexPair(bound - 1, j).IX(length)] : (*c_map)[IndexPair(bound - 1, j).IX(length)];
	}

	(*c_map)[IndexPair(0, 0).IX(length)] = ((*c_map)[IndexPair(1, 0).IX(length)] + (*c_map)[IndexPair(0, 1).IX(length)]) * .5f;
	(*c_map)[IndexPair(0, bound).IX(length)] = ((*c_map)[IndexPair(1, bound).IX(length)] + (*c_map)[IndexPair(0, bound - 1).IX(length)]) * .5f;
	(*c_map)[IndexPair(bound, 0).IX(length)] = ((*c_map)[IndexPair(bound - 1, 0).IX(length)] + (*c_map)[IndexPair(bound, 1).IX(length)]) * .5f;
	(*c_map)[IndexPair(bound, bound).IX(length)] = ((*c_map)[IndexPair(bound - 1, bound).IX(length)] + (*c_map)[IndexPair(bound, bound - 1).IX(length)]) * .5f;
}