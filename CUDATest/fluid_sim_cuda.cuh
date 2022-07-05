#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudamap.cuh"
#include "index_pair_cuda.cuh"
#include "diagnostic_statistics.cuh"

#include "vector_field.hpp"
#include "handler_methods.hpp"
#include "fluid_sim.hpp"
#include "axis_data.hpp"

#include <iostream>

__global__ void LinearSolverKernel(HashMap* data, HashMap* data_prev, float a_fac, float c_fac, uint3 length, unsigned int iter, int bounds);

cudaError_t LinearSolverCuda(int bounds, AxisData& current, AxisData& previous, const float& a_fac, const float& c_fac, const unsigned int& iter, const uint3& length);

__global__ void AdvectKernel(HashMap* data, HashMap* data_prev, HashMap* velocity_x, HashMap* velocity_y, HashMap* velocity_z, float dt, uint3 length, int bounds);

cudaError_t AdvectCuda(int bounds, AxisData& current, AxisData& previous, VectorField& velocity, const float& dt, const uint3& length);

__global__ void ProjectKernel(HashMap* velocity_x, HashMap* velocity_y, HashMap* velocity_z, HashMap* data, HashMap* data_prev, uint3 length);

__global__ void ProjectKernel2(HashMap* velocity_x, HashMap* velocity_y, HashMap* velocity_z, HashMap* data, HashMap* data_prev, uint3 length);

cudaError_t ProjectCuda(int bounds, VectorField& velocity, VectorField& velocity_prev, const uint3& length, const unsigned int& iter);

__global__ void BoundaryConditions(int bounds, HashMap* c_map, uint3 length);

cudaError_t BoundaryConditionsCuda(int bounds, HashMap* map, const uint3& length);

__global__ void AddOnAxisCuda(AxisData& data, IndexPair origin, const uint3& length);