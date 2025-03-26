//
// Created by bei on 24-12-29.
//

#include <texture_types.h>

//#define BMW
#include "gdt/math/vec.h"

namespace osc {
    using namespace gdt;

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
        int maxBounce = 10;
        float lightness_change = 0.25f;
        float contrast_change = 0.2f;
        float saturate_change = -0.2f;
    };

    struct TriangleMeshSBTData {
        vec3f color;
        vec3f *vertex;
        vec3f *normal;
        vec2f *texcoord;
        vec3i *index;
        bool hasTexture;
        cudaTextureObject_t texture;
    };
}
