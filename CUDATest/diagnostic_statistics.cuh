#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudamap.cuh"
#include "axis_data.hpp"

__global__ void Maximum(HashMap* data, uint3 length, float* max);

float MaximumCuda(AxisData& map, const uint3& length);

__global__ void Total(HashMap* data, uint3 length, float* max);

float TotalCuda(AxisData& map, const uint3& length);