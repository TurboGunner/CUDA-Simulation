#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "raypath.cuh"

#ifndef Pi
#define Pi 3.14159265358979323846
#endif

inline float3 UnitDiskRandom(curandState* rand_state) {
    float3 p;
    while (DotProduct(p, p) >= 1.0) {
        p = 2.0 * float3(curand_uniform(rand_state), curand_uniform(rand_state), 0.0f) - float3(1.0f, 1.0f, 0.0f);
    }
    return p;
}

class Camera {
public:
    Camera(float3 lookfrom, float3 lookat, float3 vup, float vfov, float aspect, float aperture, float focus_dist) { // vfov is top to bottom in degrees
        lens_radius = aperture / 2;
        float theta = vfov * Pi / 180;
        float half_height = tan(theta / 2);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = UnitVector(SubtractVector(lookfrom, lookat);
        u = UnitVector(CrossProduct(vup, w));
        v = CrossProduct(w, u);
        lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
        horizontal = 2 * half_width * focus_dist * u;
        vertical = 2 * half_height * focus_dist * v;
    }
    Ray GetRay(float s, float t) {
        float3 rd = MultiplyByScalar(UnitDiskRandom(), lens_radius);
        float3 offset = AddVector(MultiplyByScalar(u, rd.x), MultiplyByScalar(v, rd.y));
        return Ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
    }

    float3 origin;
    float3 lower_left_corner;
    float3 horizontal;
    float3 vertical;
    float3 u, v, w;
    float lens_radius;
};