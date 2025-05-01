//
// Created by bei on 24-12-29.
//

#pragma once
#include <texture_types.h>

//#define BMW
#include <Material_def.h>

#include "gdt/math/vec.h"

namespace osc {
    using namespace gdt;

    struct LightTriangle {
        vec3f v0, v1, v2;
        vec3f normal;
        vec3f emission;
        float area;
    };

    struct LaunchParams {
        struct {
            int frameID;
            float4 *colorBuffer;
            vec2i size;
            float4 *renderBuffer;
        } frame;

        struct {
            vec3f position;
            vec3f direction;
            vec3f horizontal;
            vec3f vertical;
        } camera;

        OptixTraversableHandle traversable;
        int numPixelSamples = 10;
        int maxBounce = 24;
        float lightness_change = 0.f;
        float contrast_change = 0.f;
        float saturate_change = 0.f;
        /*
        float lightness_change = 0.25f;
        float contrast_change = 0.2f;
        float saturate_change = -0.2f;
        */
        bool has_envmap = false;
        LightTriangle *lightTriangles = nullptr; // device指针
        int numLightTriangles = 0;
    };

    struct TriangleMeshSBTData {
        vec3f *vertex;
        vec3f *normal;
        vec2f *texcoord;
        vec3i *index;
        Material mat;
    };
}
