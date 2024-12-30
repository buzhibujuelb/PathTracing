//
// Created by bei on 24-12-30.
//

#include "CUDABuffer.h"
#include "LaunchParams.h"

namespace osc {
    class SampleRenderer {
    public:
        SampleRenderer();

        void render();

        void resize(const vec2i& size);

        void downloadPixels(uint32_t h_pixels[]);
    protected:
        void initOptix();

        void createContext();

        void createModule();

        void createRaygenPrograms();

        void createMissPrograms();

        void createHitgroupPrograms();

        void createPipeline();

        void buildSBT();

    protected:
        CUcontext       cuContext;
        CUstream        cuStream;
        cudaDeviceProp   cuDeviceProps;

        OptixDeviceContext optixContext;

        OptixPipeline pipeline;
        OptixPipelineCompileOptions pipelineCompileOptions = {};
        OptixPipelineLinkOptions pipelineLinkOptions = {};

        OptixModule module;
        OptixModuleCompileOptions moduleCompileOptions = {};

        std::vector<OptixProgramGroup> rayGenPrograms;
        CUDABuffer raygenRecordsBuffer;
        std::vector<OptixProgramGroup> missPrograms;
        CUDABuffer missRecordsBuffer;
        std::vector<OptixProgramGroup> hitgroupPrograms;
        CUDABuffer hitgroupRecordsBuffer;

        OptixShaderBindingTable sbt= {};

        LaunchParams launchParams;
        CUDABuffer launchParamsBuffer;

        CUDABuffer colorBuffer;
    };
}