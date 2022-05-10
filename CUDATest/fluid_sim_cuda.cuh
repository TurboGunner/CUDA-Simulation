#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_sim_helpers.cuh"

#include "vector_field.hpp"
#include "handler_methods.hpp"
#include "handler_wrapper.hpp"

#include <iostream>
#include <functional>
#include <vector>

using std::reference_wrapper;
using std::vector;
using std::function;

__global__ void LinearSolverKernel(float* data, const float* data_prev, float a_fac, float c_fac, unsigned int length, unsigned int iter, int bounds);

void LinearSolverCuda(int bounds, AxisData& current, AxisData& previous, const float& a_fac, const float& c_fac, const unsigned int& iter, const unsigned int& length);

__global__ void AdvectKernel(float* data, float* data_prev, float3* velocity, float dt, unsigned int length, int bounds);

void AdvectCuda(int bounds, AxisData& current, AxisData& previous, VectorField& velocity, const float& dt, const unsigned int& length);

__global__ void ProjectKernel(float3* velocity, float* data, float* data_prev, unsigned int length, unsigned int iter, int bounds);

void ProjectCuda(int bounds, VectorField& velocity, VectorField& velocity_prev, const unsigned int& length, const unsigned int& iter);
