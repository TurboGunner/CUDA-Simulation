#pragma once

#include "fluid_sim_cuda.cuh"

//LBM Streaming

__global__ void StreamKernel(HashMap* data, HashMap* data_prev, uint3 length);

cudaError_t StreamCuda(int bounds, AxisData& current, AxisData& previous, const uint3& length);

//LBM Advection

__global__ void LBMAdvectKernel(HashMap* data, HashMap* velocity_x, HashMap* velocity_y, HashMap* velocity_z, float visc, float total_density, float dt, float3 total_v, uint3 length);

cudaError_t LBMAdvectCuda(int bounds, AxisData& current, VectorField& velocity, const float& visc, const float& dt, const uint3& length);