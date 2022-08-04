#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "raypath.cuh"

#ifndef Pi
#define Pi 3.14159265358979323846
#endif
#include "vector_helpers.cuh"

inline Vector3D UnitDiskRandom(curandState* rand_state) {
    Vector3D p;
    while (DotProduct(p, p) >= 1.0) {
        p = Vector3D(curand_uniform(rand_state), curand_uniform(rand_state), 0.0f);
        p = MultiplyByScalar(p, 2.0f);
        p = SubtractVector(p, Vector3D(1.0f, 1.0f, 0.0f));
    }
    return p;
}

class Camera {
public:
    Camera(Vector3D lookfrom, Vector3D lookat, Vector3D vup, float vfov, float aspect, float aperture, float focus_dist) { // vfov is top to bottom in degrees
        lens_radius = aperture / 2;

        float theta = vfov * Pi / 180;
        float half_height = tan(theta / 2);
        float half_width = aspect * half_height;

        origin = lookfrom;

        w = UnitVector(SubtractVector(lookfrom, lookat));
        u = UnitVector(CrossProduct(vup, w));
        v = CrossProduct(w, u);

        Vector3D intermediate1 = MultiplyByScalar(u, half_width * focus_dist);
        Vector3D intermediate2 = MultiplyByScalar(v, (half_height * focus_dist));
        Vector3D intermediate3 = MultiplyByScalar(w, focus_dist);

        intermediate2 = SubtractVector(intermediate2, intermediate3);
        intermediate1 = SubtractVector(intermediate1, intermediate2);
        lower_left_corner = SubtractVector(origin, intermediate1);

        horizontal = MultiplyByScalar(u, half_width * focus_dist * 2.0f);
        vertical = MultiplyByScalar(v, half_height * focus_dist * 2.0f);
    }
    Ray GetRay(float s, float t, curandState* rand_state) {
        Vector3D rd = MultiplyByScalar(UnitDiskRandom(rand_state), lens_radius);
        Vector3D offset = AddVector(MultiplyByScalar(u, rd.x), MultiplyByScalar(v, rd.y));

        Vector3D intermediate1 = MultiplyByScalar(horizontal, s);
        Vector3D intermediate2 = MultiplyByScalar(vertical, t);
        Vector3D intermediate3 = SubtractVector(origin, offset);

        Vector3D ending = AddVector(lower_left_corner, AddVector(intermediate1, SubtractVector(intermediate2, intermediate3)));

        return Ray(AddVector(origin, offset), ending);
    }

    Vector3D origin;
    Vector3D lower_left_corner;
    Vector3D horizontal;
    Vector3D vertical;
    Vector3D u, v, w;

    float lens_radius;
};