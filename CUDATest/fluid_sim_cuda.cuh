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

__global__ void LinearSolverKernel(HashMap<float>* data, HashMap<float>* data_prev, float a_fac, float c_fac, uint3 length, unsigned int iter, int bounds);

void LinearSolverCuda(int bounds, AxisData& current, AxisData& previous, const float& a_fac, const float& c_fac, const unsigned int& iter, const uint3& length);

__global__ void AdvectKernel(HashMap<float>* data, HashMap<float>* data_prev, HashMap<float>* velocity_x, HashMap<float>* velocity_y, HashMap<float>* velocity_z, float dt, uint3 length, int bounds);

void AdvectCuda(int bounds, AxisData& current, AxisData& previous, VectorField& velocity, const float& dt, const uint3& length);

__global__ void ProjectKernel(HashMap<float>* velocity_x, HashMap<float>* velocity_y, HashMap<float>* velocity_z, HashMap<float>* data, HashMap<float>* data_prev, uint3 length);

__global__ void ProjectKernel2(HashMap<float>* velocity_x, HashMap<float>* velocity_y, HashMap<float>* velocity_z, HashMap<float>* data, HashMap<float>* data_prev, uint3 length);

void ProjectCuda(int bounds, VectorField& velocity, VectorField& velocity_prev, const uint3& length, const unsigned int& iter);

__global__ void BoundaryConditions(int bounds, HashMap<float>* c_map, uint3 length);

void BoundaryConditionsCuda(int bounds, HashMap<float>* map, const uint3& length);