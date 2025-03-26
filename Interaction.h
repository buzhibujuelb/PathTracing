//
// Created by bei on 25-3-26.
//

#include "gdt/math/vec.h"
#include "LaunchParams.h"
#include "math.h"

using namespace osc;

struct Ray {
    vec3f origin;
    vec3f direction;
    float tmax = FLT_MAX;
};

struct Interaction {
    float bias = 1e-3f;
    float distance = 0;
    vec3f position;
    vec3f geoNormal;
    vec3f mat_color;

    __forceinline__ __device__ Ray spawnRay(const vec3f &wi) {
        vec3f N = geoNormal;
        if (dot(wi, N) < 0) N = -N;

        Ray ray;
        ray.origin = position+N*bias;
        ray.direction = wi;
        ray.tmax = FLT_MAX;
        return ray;
    }
};