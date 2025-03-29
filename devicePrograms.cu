#include <optix_device.h>
#include <cuda_runtime.h>
#include "gdt/random/random.h"
#include "Interaction.h"
#include "PostProcess.h"

using namespace osc;

namespace osc {
    enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

    constexpr float PI = 3.1415926535897932f;

    typedef gdt::LCG<16> Random;

    struct PRD {
        Random random;
        vec3f pixelColor;
        vec3f pixelNormal;
        vec3f pixelAlbedo;
    };

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
        const TriangleMeshSBTData &sbtData = *(const TriangleMeshSBTData *) optixGetSbtDataPointer();
        Interaction &prd = *(Interaction *) getPRD<Interaction>();
        const int primID = optixGetPrimitiveIndex();
        const vec3i index = sbtData.index[primID];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;
        const vec3f &A = sbtData.vertex[index.x];
        const vec3f &B = sbtData.vertex[index.y];
        const vec3f &C = sbtData.vertex[index.z];

        vec3f N;
        if (sbtData.normal) {
            N = (1 - u - v) * sbtData.normal[index.x] + u * sbtData.normal[index.y] + v * sbtData.normal[index.z];
            if (N == vec3f(0.f)) {
                N = cross(B - A, C - A);
            }
        } else {
            N = cross(B - A, C - A);
        }

        N = normalize(N);


        const vec3f rayDir = optixGetWorldRayDirection();
        if (dot(N, rayDir) > 0) N = -N;

        prd.position = (1 - u - v) * A + u * B + v * C;
        prd.geoNormal = N;
        vec3f diffuseColor = sbtData.color;

        if (sbtData.hasTexture && sbtData.texcoord) {
            const vec2f tc = (1 - u - v) * sbtData.texcoord[index.x] + u * sbtData.texcoord[index.y] + v * sbtData.
                             texcoord[index.z];
            vec4f fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
            diffuseColor *= (vec3f) fromTexture;
        }

        float cosDN = 0.2f + .8f * fabsf(dot(rayDir, N));
        prd.mat_color = cosDN * diffuseColor;
        prd.mat_color = cosDN ;
    }

    extern "C" __global__ void __anyhit__radiance() {
    }

    static __device__ vec2f sampling_equirectangular_map(vec3f dir) {
        float phi = atan2f(dir.z, dir.x);         // [-π, π]
        float theta = acosf(clamp(dir.y, -1.f, 1.f)); // [0, π]
        float u = (phi + PI) / (2.0f * PI);     // [0, 1]
        float v = theta / PI;                     // [0, 1]
        return vec2f(u, v);
    }

    extern "C" __global__ void __miss__radiance() {
        Interaction &isec = *(Interaction *) getPRD<Interaction>();
        isec.distance = FLT_MAX;
        const cudaTextureObject_t &sbData = *(const cudaTextureObject_t *) optixGetSbtDataPointer();
        if (!sbData) {
            isec.mat_color = vec3f(1);
            return ;
        }
        vec3f rayDir = optixGetWorldRayDirection();
        vec2f uv = sampling_equirectangular_map(rayDir);
        vec4f fromTexture = tex2D<float4>(sbData, uv.x, uv.y);
        isec.mat_color = (vec3f) fromTexture;
    }

    extern "C" __global__ void __raygen__renderFrame() {
        const int ix = optixGetLaunchIndex().x;
        const int iy = optixGetLaunchIndex().y;

        const auto &camera = optixLaunchParams.camera;

        const int numPixelSamples = optixLaunchParams.numPixelSamples;

        vec3f pixelColor = 0.f;
        PRD prd;
        prd.random.init(ix + optixLaunchParams.frame.size.x * iy, optixLaunchParams.frame.frameID);

        const vec2f screen(vec2f(ix + 0.5f, iy + 0.5f) / vec2f(optixLaunchParams.frame.size));

        vec3f rayDir = normalize(
            camera.direction + (screen.x - 0.5f) * camera.horizontal + (screen.y - 0.5f) * camera.vertical);


        for (int sampleID = 0; sampleID < numPixelSamples; sampleID++) {
            Ray ray;
            ray.origin = camera.position;
            ray.direction = rayDir;
            vec3f radiance = 0.0f;
            vec3f accum = 1.0f;

            Interaction isect; // 不能放外面
            for (int bounce = 0;; bounce++) {
                if (bounce >= optixLaunchParams.maxBounce) {
                    //radiance = 0;
                    break;
                }
                uint32_t u0, u1;
                packPointer(&isect, u0, u1);
                optixTrace(optixLaunchParams.traversable,
                           ray.origin,
                           ray.direction,
                           0.f, // tmin
                           1e20f, // tmax
                           0.0f, // rayTime
                           OptixVisibilityMask(255),
                           OPTIX_RAY_FLAG_DISABLE_ANYHIT, //OPTIX_RAY_FLAG_NONE,
                           SURFACE_RAY_TYPE, // SBT offset
                           RAY_TYPE_COUNT, // SBT stride
                           SURFACE_RAY_TYPE, // missSBTIndex
                           u0, u1);
                if (isect.distance == FLT_MAX) {
                    if (bounce > 0 || !optixLaunchParams.has_envmap) {
                        radiance += isect.mat_color * accum;
                    } else {
                        radiance += isect.mat_color * accum /3;
                    }
                    break;
                }
                radiance += 0;
                accum *= isect.mat_color;
                vec3f wi;
                vec3f rnd;
                rnd.x = prd.random() * 2 - 1;
                rnd.y = prd.random() * 2 - 1;
                rnd.z = prd.random() * 2 - 1;
                wi = normalize(isect.geoNormal + normalize(rnd));
                ray = isect.spawnRay(wi);
                //printf("sample %d: radiance = %.2f %.2f %.2f\n", sampleID, radiance.x, radiance.y, radiance.z);
            }
            pixelColor += radiance;
        }
        vec4f rgba(pixelColor / numPixelSamples, 1.f);
        rgba.x = powf(rgba.x, 1 / 2.2f);
        rgba.y = powf(rgba.y, 1 / 2.2f);
        rgba.z = powf(rgba.z, 1 / 2.2f);
        if (rgba.x > 1)rgba.x = 1.0f;
        if (rgba.y > 1)rgba.y = 1.0f;
        if (rgba.z > 1)rgba.z = 1.0f;
        if (rgba.w > 1)rgba.w = 1.0f;

        // and write to frame buffer ...
        const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
        if (optixLaunchParams.frame.frameID > 0) {
            rgba += vec4f(optixLaunchParams.frame.colorBuffer[fbIndex]) * float(optixLaunchParams.frame.frameID);
            rgba /= optixLaunchParams.frame.frameID + 1.f;
        }
        optixLaunchParams.frame.colorBuffer[fbIndex] = make_float4(rgba.x, rgba.y, rgba.z, rgba.w);

        HSV hsv;
        BGR bgr;
        bgr.r = rgba.x;
        bgr.g = rgba.y;
        bgr.b = rgba.z;
        BGR2HSV(bgr, hsv);
        hsv.v += optixLaunchParams.lightness_change;
        if (hsv.s >= 0.05f)
            hsv.s += optixLaunchParams.saturate_change;
        HSV2BGR(hsv, bgr);
        Contrast(bgr, optixLaunchParams.contrast_change, 0.5f);
        rgba.x = bgr.r;
        rgba.y = bgr.g;
        rgba.z = bgr.b;
        optixLaunchParams.frame.renderBuffer[fbIndex] = make_float4(rgba.x, rgba.y, rgba.z, rgba.w);
    }
}
