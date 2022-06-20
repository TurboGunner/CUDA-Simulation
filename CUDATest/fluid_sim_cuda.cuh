#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudamap.cuh"
#include "cuda_sim_helpers.cuh"
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

__global__ void AdvectKernel(HashMap<float>* data, HashMap<float>* data_prev, HashMap<F_Vector>* velocity, float dt, unsigned int length, int bounds);

void AdvectCuda(int bounds, AxisData& current, AxisData& previous, VectorField& velocity, const float& dt, const unsigned int& length);

__global__ void ProjectKernel(HashMap<F_Vector>* velocity, HashMap<F_Vector>* velocity_output, HashMap<float>* data, HashMap<float>* data_prev, unsigned int length, unsigned int iter, int bounds);

void ProjectCuda(int bounds, VectorField& velocity, VectorField& velocity_prev, const unsigned int& length, const unsigned int& iter);

extern __device__ IndexPair* index_ptr;

inline void IndexAllocator(int size) {
	const unsigned int alloc_size = size * size;
	IndexPair* host_ptr = new IndexPair[alloc_size];

	cudaMalloc(&index_ptr, alloc_size * sizeof(IndexPair));

	unsigned int y_current = 0;
	for (y_current; y_current < size; y_current++) {
		for (unsigned int i = 0; i < size; i++) {
			IndexPair current(i, y_current);
			host_ptr[current.IX(size)] = current;
		}
	}
	cudaError_t cuda_status = cudaSuccess;
	cuda_status = CopyFunction("CachedIndexPair", index_ptr, host_ptr, cudaMemcpyHostToDevice, cuda_status, alloc_size, sizeof(IndexPair));
}