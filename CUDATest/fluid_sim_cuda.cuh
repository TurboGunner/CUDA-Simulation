#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_sim_helpers.cuh"
#include "vector_field.hpp"
#include "handler_methods.hpp"

#include <iostream>
#include <functional>
#include <vector>

__global__ void LinearSolverKernel(float* result_ptr, float* data, const float* data_prev, float a_fac, float c_fac, unsigned int length, unsigned int iter);

float* LinearSolverCuda(int bounds, VectorField& current, VectorField& previous, const float& a_fac, const float& c_fac, const unsigned int& iter, const unsigned int& length);

__global__ void AdvectKernel(float* result_ptr, float* data, float* data_prev, float3* velocity, float dt, unsigned int length);

float* AdvectCuda(int bounds, VectorField& current, VectorField& previous, VectorField& velocity, const float& dt, const unsigned int& length);