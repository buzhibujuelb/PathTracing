#include <optix_device.h>
#include "LaunchParams.h"

using namespace osc;

namespace osc{
    extern "C" __constant__ LaunchParams optixLaunchParams;

    extern "C" __global__ void __closesthit__radiance(){
    }

    extern "C" __global__ void __anyhit__radiance(){
    }

    extern "C" __global__ void __miss__radiance(){
    }

    extern "C" __global__ void __raygen__renderFrame(){
      if (optixLaunchParams.frameID == 0 && optixGetLaunchIndex().x ==0 && optixGetLaunchIndex().y ==0){
        printf("###########################################\n");
        printf("Hello world from OptiX 7 raygen program!\n(within a %ix%i-sized launch)\n", optixLaunchParams.fbSize.x, optixLaunchParams.fbSize.y);
        printf("###########################################\n");
      }
      const int ix = optixGetLaunchIndex().x;
      const int iy = optixGetLaunchIndex().y;

      const int r = ix % 256;
      const int g = iy % 256;
      const int b = (ix+iy) % 256;
      const uint32_t rgba = 0xff000000 | (b << 16) | (g << 8) | r;

      const uint32_t fbIndex = ix+iy*optixLaunchParams.fbSize.x;
      optixLaunchParams.colorBuffer[fbIndex] = rgba;

    }
}