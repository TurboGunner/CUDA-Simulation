#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "raypath.cuh"
#include "vector_helpers.cuh"
#include "vector_helpers.cuh"

struct Ray {
	__device__ Ray() = default;
	__device__ Ray(const float3& start_in, const float3& end_in) {
		start = start_in;
		end = end_in;
	}

	__device__ float3 Origin() const {
		return start;
	}

	__device__ float3 Direction() const {
		return end;
	}

	__device__ float3 PointTowards(const float& t) const {
		float3 multiplier = MultiplyByScalar(end, t);

		multiplier.x += start.x;
		multiplier.y += start.y;
		multiplier.z += start.z;

		return multiplier;
	}

	float3 start, end;
};

struct RayHit {
	float t;
	float3 p;
	float3 normal;
	Material* mat_ptr;
};

class Hitable {
public:
	virtual bool Hit(const Ray& Ray, float t_min, float t_max, RayHit& hit) const = 0;
};

class HitableList : public Hitable {
public:
	HitableList() = default;

	HitableList(Hitable** l, int n) {
		list = l; list_size = n;
	}

	virtual bool Hit(const Ray& Ray, float t_min, float t_max, RayHit& hit) const;

	Hitable** list;
	int list_size;
};

bool HitableList::Hit(const Ray& Ray, float t_min, float t_max, RayHit& hit) const {
	RayHit temp_hit;
	bool successful_hit = false;
	double closest_so_far = t_max;
	for (int i = 0; i < list_size; i++) {
		if (list[i]->Hit(Ray, t_min, closest_so_far, temp_hit)) {
			successful_hit = true;
			closest_so_far = temp_hit.t;
			hit = temp_hit;
		}
	}
	return successful_hit;
}