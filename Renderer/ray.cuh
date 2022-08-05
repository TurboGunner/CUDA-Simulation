#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

class Material;

#include "raypath.cuh"
#include "material.cuh"

struct RayHit {
	__host__ __device__ RayHit() {};
	float t;
	Vector3D p;
	Vector3D normal;
	Material* mat_ptr;
};

struct Ray {
	__host__ __device__ Ray() { };

	__host__ __device__ Ray(const Vector3D& start_in, const Vector3D& end_in) {
		start = start_in;
		end = end_in;
	}

	__host__ __device__ Vector3D Origin() const {
		return start;
	}

	__host__ __device__ Vector3D Direction() const {
		return end;
	}

	__host__ __device__ Vector3D PointTowards(const float& t) const {
		Vector3D multiplier = MultiplyByScalar(end, t);

		multiplier.dim[0] += start.x();
		multiplier.dim[1] += start.y();
		multiplier.dim[2] += start.z();

		return multiplier;
	}

	Vector3D start, end;
};