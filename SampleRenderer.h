//
// Created by bei on 24-12-30.
//

#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "Model.h"
#include "gdt/math/AffineSpace.h"

namespace osc {

    struct Camera {
        vec3f from;
        vec3f at;
        vec3f up;
    };

    class SampleRenderer {
    public:
        SampleRenderer(const Model* model);

        void render();

        void resize(const vec2i& size);

        void downloadPixels(uint32_t h_pixels[]);

        void setCamera(const Camera& camera);

        void createTextures();

    protected:
        void initOptix();

        void createContext();

        void createModule();

        void createRaygenPrograms();

        void createMissPrograms();

        void createHitgroupPrograms();

        void createPipeline();

        void buildSBT();

        OptixTraversableHandle buildAccel();

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

        Camera lastSetCamera;

        const Model *model;
        std::vector<CUDABuffer> vertexBuffer;
        std::vector<CUDABuffer> indexBuffer;
        std::vector<CUDABuffer> normalBuffer;
        std::vector<CUDABuffer> texcoordBuffer;
        //! buffer that keeps the (final, compacted) accel structure
        CUDABuffer asBuffer;

        std::vector<cudaArray_t> textureArrays;
        std::vector<cudaTextureObject_t> textureObjects;
    };

}