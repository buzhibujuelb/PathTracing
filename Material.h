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

__forceinline__ __device__ float randMinus1To1(Random &rng) {
    return rng() * 2.f - 1.f;
}

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

    rnd.x = randMinus1To1(prd.random);
    rnd.y = randMinus1To1(prd.random);
    rnd.z = randMinus1To1(prd.random);
    wo = normalize(isect.geoNormal + normalize(rnd));
    pdf = 1 / (2 * float(PI));
    //printf("??? return %.3f %.3f %.3f\n",bsdf.x, bsdf.y, bsdf.z);
    return bsdf;
}

__forceinline__ __device__ vec3f cal_metal_bsdf(const Interaction &isect, const vec3f &wi, vec3f &wo, float &pdf,
                                                const int ix, const int iy, const int frame_id) {
    vec3f diffuseColor = isect.mat.diffuse;
    if (isect.mat.diffuseTextureID != -1) {
        float u = isect.texcoord.x;
        float v = isect.texcoord.y;
        vec4f fromTexture = tex2D<float4>(isect.mat.diffuseTexture, u, v);
        diffuseColor *= (vec3f) fromTexture;
    }
    PRD prd;
    uint64_t seed = ((uint64_t) (ix) * 1973 + (uint64_t) (iy) * 9277 + frame_id * 26699) | 1;
    prd.random.init(seed, seed ^ 0xdeadbeef);

    vec3f out = wi - 2 * dot(wi, isect.geoNormal) * isect.geoNormal;
    vec3f out1 = cross(out, isect.geoNormal);
    vec3f out2 = cross(out, out1);
    vec3f out3 = normalize(
        out + isect.roughness * (out1 * randMinus1To1(prd.random) + out2 * randMinus1To1(prd.random)));
    if (dot(out3, isect.geoNormal) < 0) {
        out3 = -out3;
    }
    wo = out3;
    vec3f bsdf = diffuseColor / float(PI);
    pdf = 1 / (2 * float(PI));
    //printf("??? return %.3f %.3f %.3f\n",bsdf.x, bsdf.y, bsdf.z);
    return bsdf;
}

__forceinline__ __device__ vec3f cal_bsdf(const Interaction &isect, const vec3f &wi, vec3f &wo, float &pdf,
                                          const int ix, const int iy, const int frame_id) {
    vec3f result;
    switch (isect.mat.type) {
        case DIFFUSE:
            result = cal_diffuse_bsdf(isect, wi, wo, pdf, ix, iy, frame_id);
            break;
        case METAL:
            result = cal_metal_bsdf(isect, wi, wo, pdf, ix, iy, frame_id);
            break;
        default:
            printf("No MAT TYPE ERROR!\n");
            return vec3f(1);
    }
    return result;
}
