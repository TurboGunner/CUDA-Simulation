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

__global__ void ProjectKernel(HashMap<F_Vector>* velocity, HashMap<float>* data, HashMap<float>* data_prev, unsigned int length, int bounds);

__global__ void ProjectKernel2(HashMap<F_Vector>* velocity, HashMap<F_Vector>* velocity_output, HashMap<float>* data, HashMap<float>* data_prev, unsigned int length, int bounds);

void ProjectCuda(int bounds, VectorField& velocity, VectorField& velocity_prev, const unsigned int& length, const unsigned int& iter);