//
// Created by bei on 25-4-29.
//

#pragma once

#include "Interaction.h"

using namespace osc;

namespace osc {
    extern __device__ __constant__ float PI;
}

typedef gdt::LCG<16> Random;

struct PRD {
    Random random;
    //vec3f pixelColor;
};

__forceinline__ __device__ vec3f cal_diffuse_bsdf(const Interaction &isect, const vec3f &wi, vec3f &wo, float &pdf,
                                                  const int ix, const int iy, const int frame_id) {
    vec3f diffuseColor = isect.mat.diffuse;
    if (isect.mat.diffuseTextureID != -1) {
        float u = isect.texcoord.x;
        float v = isect.texcoord.y;
        vec4f fromTexture = tex2D<float4>(isect.mat.diffuseTexture, u, v);
        diffuseColor *= (vec3f) fromTexture;
    }
    vec3f bsdf = diffuseColor / float(PI);
    vec3f rnd;
    PRD prd;
    uint64_t seed = ((uint64_t) (ix) * 1973 + (uint64_t) (iy) * 9277 + frame_id * 26699) | 1;
    prd.random.init(seed, seed ^ 0xdeadbeef);

    rnd.x = prd.random() * 2 - 1;
    rnd.y = prd.random() * 2 - 1;
    rnd.z = prd.random() * 2 - 1;
    wo = normalize(isect.geoNormal + normalize(rnd));
    pdf = 1 / (2 * float(PI));
    //printf("??? return %.3f %.3f %.3f\n",bsdf.x, bsdf.y, bsdf.z);
    return bsdf;
}

__forceinline__ __device__ vec3f cal_bsdf(const Interaction &isect, const vec3f &wi, vec3f &wo, float &pdf,
                                          const int ix, const int iy, const int frame_id) {
    vec3f result;
    if (isect.mat.type == DIFFUSE) {
        result = cal_diffuse_bsdf(isect, wi, wo, pdf, ix, iy, frame_id);
    } else {
        printf("No MAT TYPE ERROR!\n");
        return vec3f(1);
    }
    return result;
}
