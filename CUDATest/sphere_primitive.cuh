#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "raypath.cuh"
#include "rayhit_logic.cuh"
#include "vector_helpers.cuh"
#include "material.cuh"

class Sphere : public Hitable {
public:
    Sphere() = default;

    __host__ __device__ Sphere(float3 cen, float r, Material* m) : center(cen), radius(r), mat_ptr(m) {};

    __host__ __device__ virtual bool Hit(const Ray& Ray, float tmin, float tmax, RayHit& hit) const;

    float3 center;
    float radius;
    Material* mat_ptr;
};

__host__ __device__ inline void AssignOnHit(RayHit& hit, const float& t, float3 point, float3 normal, Material* mat) {
    hit.t = t;
    hit.p = point;
    hit.normal = normal;
    hit.mat_ptr = mat;
}

__host__ __device__ bool Sphere::Hit(const Ray& Ray, float t_min, float t_max, RayHit& hit) const {
    float3 oc = SubtractVector(Ray.Origin(), center);

    float a = DotProduct(Ray.Direction(), Ray.Direction());
    float b = DotProduct(oc, Ray.Direction());
    float c = DotProduct(oc, oc) - radius * radius;

    float discriminant = b * b - a * c;

    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant)) / a;
        float3 normal = DivideByScalar(SubtractVector(hit.p, center), radius);
        if (temp < t_max && temp > t_min) {
            AssignOnHit(hit, temp, Ray.PointTowards(hit.t), normal, mat_ptr);
            return true;
        }

        temp = (-b + sqrt(discriminant)) / a;

        if (temp < t_max && temp > t_min) {
            AssignOnHit(hit, temp, Ray.PointTowards(hit.t), normal, mat_ptr);
            return true;
        }
    }
    return false;
}