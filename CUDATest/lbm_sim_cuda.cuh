#pragma once

#include "fluid_sim_cuda.cuh"

//LBM Streaming

__global__ void StreamKernel(HashMap* data, uint3 length);

cudaError_t StreamCuda(int bounds, AxisData& current, const uint3& length);

//LBM Advection

__device__ float EquilibriumFunction(HashMap* velocity_x, HashMap* velocity_y, HashMap* velocity_z, float total_density, IndexPair incident, float* total_v, uint3 length);

__global__ void LBMAdvectKernel(HashMap* data, HashMap* velocity_x, HashMap* velocity_y, HashMap* velocity_z, float* total_v, float visc, float total_density, float dt, uint3 length);

cudaError_t LBMAdvectCuda(AxisData& current, VectorField& velocity, const float& visc, const float& dt, const uint3& length);

//General Operation

__global__ void VectorNormalKernel(HashMap* velocity_x, HashMap* velocity_y, HashMap* velocity_z, uint3 length);

cudaError_t VectorNormalCuda(VectorField& velocity, const uint3& length);