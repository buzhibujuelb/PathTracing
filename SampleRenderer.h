//
// Created by bei on 24-12-30.
//

#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "gdt/math/AffineSpace.h"

namespace osc {

    struct Camera {
        vec3f from;
        vec3f at;
        vec3f up;
    };

    struct TriangleMesh {
        /*! add a unit cube (subject to given xfm matrix) to the current
            triangleMesh */
        void addUnitCube(const affine3f &xfm);

        //! add aligned cube aith front-lower-left corner and size
        void addCube(const vec3f &center, const vec3f &size);

        std::vector<vec3f> vertex;
        std::vector<vec3i> index;
      };

    class SampleRenderer {
    public:
        SampleRenderer(const TriangleMesh &model);

        void render();

        void resize(const vec2i& size);

        void downloadPixels(uint32_t h_pixels[]);

        void setCamera(const Camera& camera);

    protected:
        void initOptix();

        void createContext();

        void createModule();

        void createRaygenPrograms();

        void createMissPrograms();

        void createHitgroupPrograms();

        void createPipeline();

        void buildSBT();

        OptixTraversableHandle buildAccel(const TriangleMesh &model);

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

        const TriangleMesh model;
        CUDABuffer vertexBuffer;
        CUDABuffer indexBuffer;
        //! buffer that keeps the (final, compacted) accel structure
        CUDABuffer asBuffer;
    };

}