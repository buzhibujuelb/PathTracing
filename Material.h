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

struct LightSample {
    vec3f position;
    vec3f normal;
    vec3f emission;
    float pdf;
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

__forceinline__ __device__ float my_min(const float a, const float b) {
    return a < b ? a : b;
}

__forceinline__ __device__ float length_squared(const vec3f v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

__forceinline__ __device__ vec3f reflect(const vec3f v, const vec3f n) {
    return v - 2 * dot(v, n) * n;
}

__forceinline__ __device__ vec3f refract(const vec3f uv, const vec3f n, double etai_over_etat) {
    auto cos_theta = dot(-uv, n);
    vec3f r_out_perp = (float) etai_over_etat * (uv + cos_theta * n);
    vec3f r_out_parallel = (float) (-sqrt(fabs(1.0 - length_squared(r_out_perp)))) * n;
    return r_out_perp + r_out_parallel;
}

__forceinline__ __device__ double schlick(double cosine, double ref_idx) {
    auto r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 *= r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}

__forceinline__ __device__ vec3f cal_dielectric_bsdf(const Interaction &isect, const vec3f &wi, vec3f &wo, float &pdf,
                                                     const int ix, const int iy, const int frame_id) {
    vec3f diffuseColor = isect.mat.diffuse;
    if (isect.mat.diffuseTextureID != -1) {
        float u = isect.texcoord.x;
        float v = isect.texcoord.y;
        vec4f fromTexture = tex2D<float4>(isect.mat.diffuseTexture, u, v);
        diffuseColor *= (vec3f) fromTexture;
    }
    vec3f bsdf = diffuseColor;
    pdf = 1;
    vec3f out;
    float etai_over_etat = 0;
    if (dot(wi, isect.realNormal) > 0) etai_over_etat = isect.mat.ior;
    else etai_over_etat = 1.0f / isect.mat.ior;
    vec3f unit_direction = normalize(wi);
    double cos_theta = my_min(dot(-unit_direction, isect.geoNormal), 1.0);
    double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    if (etai_over_etat * sin_theta > 1.0f) {
        //全内反射
        wo = reflect(unit_direction, isect.geoNormal);
        return bsdf;
    }
    double reflect_prob = schlick(cos_theta, etai_over_etat); //反射率
    PRD prd;
    prd.random.init(frame_id * 234834 % 32849 + ix * 385932 % 82921, frame_id * 348593 % 43832 + iy * 324123 % 23415);
    if (prd.random() < reflect_prob) {
        wo = reflect(unit_direction, isect.geoNormal);
        return bsdf;
    }
    if (prd.random() < isect.mat.transparent) {
        //wo = reflect(unit_direction, isect.geomNormal);
        //return bsdf;
    }
    wo = refract(unit_direction, isect.geoNormal, etai_over_etat);
    return bsdf;
}

__forceinline__ __device__ vec3f cal_bsdf(const Interaction &isect, const vec3f &wi, vec3f &wo, float &pdf,
                                          const int ix, const int iy, const int frame_id) {
    vec3f result;
    PRD prd;
    uint64_t seed = ((uint64_t) (ix) * 1973 + (uint64_t) (iy) * 9277 + frame_id * 26699) | 1;
    switch (isect.mat.type) {
        case DIFFUSE:
            result = cal_diffuse_bsdf(isect, wi, wo, pdf, ix, iy, frame_id);
            break;
        case METAL:
            prd.random.init(seed, seed ^ 0xdeadbeef);
            if ((float) prd.random() < isect.mat.roughness)
                result = cal_diffuse_bsdf(isect, wi, wo, pdf, ix, iy, frame_id);
            else
                result = cal_metal_bsdf(isect, wi, wo, pdf, ix, iy, frame_id);
            break;
        case DIELECTRIC:
            result = cal_dielectric_bsdf(isect, wi, wo, pdf, ix, iy, frame_id);
            break;
        default:
            printf("No MAT TYPE ERROR!\n");
            return vec3f(1);
    }
    return result;
}

__device__ LightSample sample_light(const LightTriangle *lightTriangles, int numLightTriangles, Random &rng) {
    if (numLightTriangles == 0) return {};
    int triID = int(rng() * numLightTriangles);
    const LightTriangle &tri = lightTriangles[triID];
    // 面内均匀采样
    float u = rng(), v = rng();
    if (u + v > 1.0f) {
        u = 1 - u;
        v = 1 - v;
    }
    vec3f pos = tri.v0 * (1 - u - v) + tri.v1 * u + tri.v2 * v;
    float pdf = 1.0f / (tri.area * numLightTriangles);
    return {pos, tri.normal, tri.emission, pdf};
}
