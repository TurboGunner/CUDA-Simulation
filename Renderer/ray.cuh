#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "raypath.cuh"

struct Ray {
	__device__ Ray() { };

	__device__ Ray(const Vector3D& start_in, const Vector3D& end_in) {
		start = start_in;
		end = end_in;
	}

	__device__ Vector3D Origin() const {
		return start;
	}

	__device__ Vector3D Direction() const {
		return end;
	}

	__device__ Vector3D PointTowards(const float& t) const {
		Vector3D multiplier = MultiplyByScalar(end, t);

		multiplier.x += start.x;
		multiplier.y += start.y;
		multiplier.z += start.z;

		return multiplier;
	}

	Vector3D start, end;
};