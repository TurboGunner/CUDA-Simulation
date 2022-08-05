#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "raypath.cuh"
#include "vector_helpers.cuh"
#include "material.cuh"
#include "ray.cuh"

class Hitable {
public:
	__device__ virtual bool Hit(const Ray& Ray, float t_min, float t_max, RayHit& hit) const = 0;
};