#pragma once

#include "fluid_sim_cuda.cuh"

//LBM Streaming

__global__ void StreamKernel(HashMap* data, uint3 length);

cudaError_t StreamCuda(int bounds, AxisData& current, const uint3& length);

//LBM Advection

__global__ void LBMAdvectKernel(HashMap* data, HashMap* velocity_x, HashMap* velocity_y, HashMap* velocity_z, float* total_v, float visc, float total_density, float dt, uint3 length);

cudaError_t LBMAdvectCuda(int bounds, AxisData& current, VectorField& velocity, const float& visc, const float& dt, const uint3& length);