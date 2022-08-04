#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "raypath.cuh"
#include "ray.cuh"
#include "vector_helpers.cuh"
#include "rayhit_logic.cuh"

__device__ Vector3D RandomVector(curandState* rand_state) {
    return Vector3D(curand_uniform(rand_state), curand_uniform(rand_state), curand_uniform(rand_state));
}

class Material {
public:
    __device__ virtual bool Scatter(const Ray& r_in, const RayHit& hit, Vector3D& attenuation, Ray& scatter, curandState* rand_state) const = 0;
};

__device__ Vector3D UnitSphereRandom(curandState* rand_state) {
    Vector3D p;
    while(SquaredLength(p) >= 1.0f) {
        Vector3D random_intermediate = MultiplyByScalar(RandomVector(rand_state), 2.0f);
        p = SubtractByScalar(random_intermediate, 1.0f);
    }
    return p;
}


class Lambertian : public Material {
public:
    __host__ __device__ Lambertian(const Vector3D& a) {
        albedo = a;
    }
    __device__ virtual bool Scatter(const Ray& r_in, const RayHit& hit, Vector3D& attenuation, Ray& scatter, curandState* rand_state) const {
        Vector3D target = AddVector(AddVector(hit.p, hit.normal), UnitSphereRandom(rand_state));
        scatter = Ray(hit.p, SubtractVector(target, hit.p));
        attenuation = albedo;
        return true;
    }

    Vector3D albedo;
};