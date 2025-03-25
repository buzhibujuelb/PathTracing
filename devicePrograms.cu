#include <optix_device.h>
#include <cuda_runtime.h>
#include "LaunchParams.h"

using namespace osc;

namespace osc {
    enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

    static __forceinline__ __device__
    void *unpackPointer(uint32_t i0, uint32_t i1) {
        const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
        void *ptr = reinterpret_cast<void *>(uptr);
        return ptr;
    }

    static __forceinline__ __device__
    void packPointer(void *ptr, uint32_t &i0, uint32_t &i1) {
        const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
        i0 = uptr >> 32;
        i1 = uptr & 0x00000000ffffffff;
    }

    template<typename T>
    static __forceinline__ __device__ T *getPRD() {
        const uint32_t u0 = optixGetPayload_0();
        const uint32_t u1 = optixGetPayload_1();
        return reinterpret_cast<T *>(unpackPointer(u0, u1));
    }

    extern "C" __constant__ LaunchParams optixLaunchParams;

    extern "C" __global__ void __closesthit__radiance() {
        const TriangleMeshSBTData &sbtData = *(const TriangleMeshSBTData *)optixGetSbtDataPointer();
        const int primID = optixGetPrimitiveIndex();
        const vec3i index = sbtData.index[primID];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;
        const vec3f &A = sbtData.vertex[index.x];
        const vec3f &B = sbtData.vertex[index.y];
        const vec3f &C = sbtData.vertex[index.z];

        vec3f N;
        if (sbtData.normal) {
            N = normalize((1-u-v)*sbtData.normal[index.x] + u*sbtData.normal[index.y] + v*sbtData.normal[index.z]);
        } else {
            N = normalize(cross(B-A, C-A));
        }

        vec3f diffuseColor = sbtData.color;

        if (sbtData.hasTexture && sbtData.texcoord) {
            const vec2f tc = (1-u-v)*sbtData.texcoord[index.x] + u*sbtData.texcoord[index.y] + v*sbtData.texcoord[index.z];
            vec4f fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
            diffuseColor*= (vec3f)fromTexture;
        }

        const vec3f rayDir = optixGetWorldRayDirection();
        const float cosDN = 0.2f + .8f* fabsf(dot(rayDir, N));
        vec3f &prd = *(vec3f *)getPRD<vec3f>();
        prd = cosDN * diffuseColor;
    }

    extern "C" __global__ void __anyhit__radiance() {
    }

    extern "C" __global__ void __miss__radiance() {
        vec3f &prd = *(vec3f *)getPRD<vec3f>();
        prd = vec3f(1.f);
    }

    extern "C" __global__ void __raygen__renderFrame() {
        const int ix = optixGetLaunchIndex().x;
        const int iy = optixGetLaunchIndex().y;

        const auto &camera = optixLaunchParams.camera;

        vec3f pixelColorPRD = vec3f(0.f);

        uint32_t u0, u1;
        packPointer(&pixelColorPRD, u0, u1);

        const vec2f screen(vec2f(ix+0.5f, iy+0.5f)/vec2f(optixLaunchParams.frame.size));

        vec3f rayDir = normalize(camera.direction+(screen.x-0.5f)*camera.horizontal+(screen.y-0.5f)*camera.vertical);

        optixTrace(optixLaunchParams.traversable,
                       camera.position,
                       rayDir,
                       0.f,    // tmin
                       1e20f,  // tmax
                       0.0f,   // rayTime
                       OptixVisibilityMask( 255 ),
                       OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
                       SURFACE_RAY_TYPE,             // SBT offset
                       RAY_TYPE_COUNT,               // SBT stride
                       SURFACE_RAY_TYPE,             // missSBTIndex
                       u0, u1 );

        const int r = int(255.99f*pixelColorPRD.x);
        const int g = int(255.99f*pixelColorPRD.y);
        const int b = int(255.99f*pixelColorPRD.z);
        // convert to 32-bit rgba value (we explicitly set alpha to 0xff
        // to make stb_image_write happy ...
        const uint32_t rgba = 0xff000000
          | (r<<0) | (g<<8) | (b<<16);
        // and write to frame buffer ...
        const uint32_t fbIndex = ix+iy*optixLaunchParams.frame.size.x;
        optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
    }
}
