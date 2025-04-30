#pragma once
#include <gdt/math/vec.h>
#include <cuda_runtime.h>

using namespace gdt;

enum MaterialType {
    DIFFUSE,
    METAL,
    DIELECTRIC,
};

struct Material {
    MaterialType type;
    vec3f diffuse;
    float ior = 1.1f;
    float transparent;
    vec3f emitter = 0;
    int diffuseTextureID = -1;
    cudaTextureObject_t diffuseTexture;
    float roughness;
};
