#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "raypath.cuh"

class Sphere : public Hitable {
public:
    __device__ Sphere() {};

    __device__ Sphere(Vector3D cen, float r, Material* m) {
        center = cen;
        radius = r;
        mat_ptr = m;
    }

    __device__ virtual bool Hit(const Ray& ray, float t_min, float t_max, RayHit& hit) const;

    __device__ void AssignOnHit(RayHit& hit, const float& t, Vector3D point, Vector3D normal, Material* mat) const { //Good
        hit.t = t;
        hit.p = point;
        hit.normal = normal;
        hit.mat_ptr = mat;
    }

    Vector3D center;
    float radius;
    Material* mat_ptr;
};

__device__ bool Sphere::Hit(const Ray& ray, float t_min, float t_max, RayHit& hit) const { //Good!
    Vector3D oc = SubtractVector(ray.Origin(), center);

    float a = DotProduct(ray.Direction(), ray.Direction());
    float b = DotProduct(oc, ray.Direction());
    float c = DotProduct(oc, oc) - radius * radius;

    float discriminant = b * b - a * c;

    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant)) / a;
        Vector3D normal = DivideByScalar(SubtractVector(hit.p, center), radius);
        if (temp < t_max && temp > t_min) {
            AssignOnHit(hit, temp, ray.PointTowards(hit.t), normal, mat_ptr);
            return true;
        }

        temp = (-b + sqrt(discriminant)) / a;

        if (temp < t_max && temp > t_min) {
            AssignOnHit(hit, temp, ray.PointTowards(hit.t), normal, mat_ptr);
            return true;
        }
    }
    return false;
}