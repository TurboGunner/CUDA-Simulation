#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <curand.h>

#include "raypath.cuh"
#include "rayhit_logic.cuh"
#include "sphere_primitive.cuh"
#include "vector_helpers.cuh"

class Material {
public:
	virtual bool Scatter(const Ray& Ray, const RayHit& hit, float3& attenuation, Ray& scatter) const = 0;
};

__host__ __device__ float3 UnitSphereRandom() {
    float3 p;
    while(SquaredLength(p) >= 1.0f) {
        float3 random_intermediate = MultiplyByScalar(float3(rand(), rand(), rand()), 2.0f);
        p = SubtractByScalar(random_intermediate, 1.0f);
    }
    return p;
}


class Lambertian : public Material {
public:
    Lambertian(const float3& a) : albedo(a) {}
    virtual bool scatter(const Ray& r_in, const RayHit& hit, float3& attenuation, Ray& scatter) const {
        float3 target = AddVector(AddVector(hit.p, hit.normal), UnitSphereRandom());
        scatter = Ray(hit.p, SubtractVector(target, hit.p));
        attenuation = albedo;
        return true;
    }

    float3 albedo;
};