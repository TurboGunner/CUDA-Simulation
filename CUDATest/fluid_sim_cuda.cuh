#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vector_field.hpp"

#include <map>

__device__ int IX(unsigned int x, unsigned int y, const unsigned int& size);

__global__ void LinearSolverKernel(float* result_ptr, const float* data, const float* data_prev, float a_fac, float c_fac, unsigned int length);

float* LinearSolverCuda(int bounds, VectorField& current, VectorField& previous, const float& a_fac, const float& c_fac);