#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "raypath.cuh"
#include "vector_helpers.cuh"

struct RayHit {
	float t;
	Vector3D p;
	Vector3D normal;
	Material* mat_ptr;
};

class Hitable {
public:
	virtual bool Hit(const Ray& Ray, float t_min, float t_max, RayHit& hit) const = 0;
};

class HitableList : public Hitable {
public:
	__device__ HitableList() = default;

	__device__ HitableList(Hitable* l, int n) {
		list = l; list_size = n;
	}

	__device__ virtual bool Hit(const Ray& ray, float t_min, float t_max, RayHit& hit) const;

	Hitable* list;
	size_t list_size;
};

__device__ bool HitableList::Hit(const Ray& ray, float t_min, float t_max, RayHit& hit) const {
	RayHit temp_hit;
	bool successful_hit = false;
	double closest_so_far = t_max;
	for (size_t i = 0; i < list_size; i++) {
		if (list[i].Hit(ray, t_min, closest_so_far, temp_hit)) {
			successful_hit = true;
			closest_so_far = temp_hit.t;
			hit = temp_hit;
		}
	}
	return successful_hit;
}