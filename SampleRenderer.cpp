//
// Created by bei on 24-12-30.
//

#include "SampleRenderer.h"
#include<optix_function_table_definition.h>

/*! \namespace osc - Optix Siggraph Course */
namespace osc {
  extern "C" char embedded_ptx_code[];
  /*! SBT record for a raygen program */
  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void *data;
  };
  /*! SBT record for a miss program */
  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void *data;
  };
  /*! SBT record for a hitgroup program */
  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    int objectID;
  };

  SampleRenderer::SampleRenderer() {
    initOptix();

    std::cout << "#osc: creating optix context ..." << std::endl;
    createContext();

    std::cout << "#osc: setting up module ..." << std::endl;
    createModule();

    std::cout << "#osc: creating raygen programs ..." << std::endl;
    createRaygenPrograms();

    std::cout << "#osc: creating miss programs ..." << std::endl;
    createMissPrograms();

    std::cout << "#osc: creating hitgroup programs ..." << std::endl;
    createHitgroupPrograms();

    std::cout << "#osc: setting up optix pipeline ..." << std::endl;
    createPipeline();

    std::cout << "#osc: building SBT ..." << std::endl;
    buildSBT();

    launchParamsBuffer.alloc(sizeof(launchParams));
    std::cout << "#osc: context, module, pipeline, etc, all set up ..." << std::endl;

    std::cout << GDT_TERMINAL_GREEN;
    std::cout << "#osc: Optix 7 Sample fully set up" << std::endl;
    std::cout << GDT_TERMINAL_DEFAULT;

  }

  void SampleRenderer::initOptix() {

 std::cout << "#osc: initializing optix..." << std::endl;
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
      throw std::runtime_error("#osc: no CUDA capable devices found!");
    std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;
    OPTIX_CHECK( optixInit() );
    std::cout << GDT_TERMINAL_GREEN
              << "#osc: successfully initialized optix... yay!"
              << GDT_TERMINAL_DEFAULT << std::endl;
  }

  static void context_log_cb(unsigned int level,
                             const char *tag,
                             const char *message,
                             void *) {
    fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
  }

  void SampleRenderer::createContext() {

    // for this sample, do everything on one device
    const int deviceID = 0;
    CUDA_CHECK(SetDevice(deviceID));
    CUDA_CHECK(StreamCreate(&cuStream));

    cudaGetDeviceProperties(&cuDeviceProps, deviceID);
    std::cout << "#osc: running on device: " << cuDeviceProps.name << std::endl;

    CUresult  cuRes = cuCtxGetCurrent(&cuContext);
    if( cuRes != CUDA_SUCCESS )
      fprintf( stderr, "Error querying current context: error code %d\n", cuRes );

    OPTIX_CHECK(optixDeviceContextCreate(cuContext, 0, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback
                (optixContext,context_log_cb,nullptr,4));
  }

void SampleRenderer::createModule() {
    moduleCompileOptions.maxRegisterCount  = 50;
    moduleCompileOptions.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur     = false;
    pipelineCompileOptions.numPayloadValues   = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags     = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

    pipelineLinkOptions.maxTraceDepth          = 2;

    const std::string ptxCode = embedded_ptx_code;

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixModuleCreate(optixContext,
                                         &moduleCompileOptions,
                                         &pipelineCompileOptions,
                                         ptxCode.c_str(),
                                         ptxCode.size(),
                                         log,&sizeof_log,
                                         &module
                                         ));
    if (sizeof_log > 1) PRINT(log);
  }

  void SampleRenderer::createRaygenPrograms() {
    // we do a single ray gen program in this example:
    rayGenPrograms.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module            = module;
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &rayGenPrograms[0]
                                        ));
    if (sizeof_log > 1) PRINT(log);
  }

  /*! does all setup for the miss program(s) we are going to use */
  void SampleRenderer::createMissPrograms() {
    // we do a single ray gen program in this example:
    missPrograms.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module            = module;
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &missPrograms[0]
                                        ));
    if (sizeof_log > 1) PRINT(log);
  }

  /*! does all setup for the hitgroup program(s) we are going to use */
  void SampleRenderer::createHitgroupPrograms() {
    // for this simple example, we set up a single hit group
    hitgroupPrograms.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH            = module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.moduleAH            = module;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &hitgroupPrograms[0]
                                        ));
    if (sizeof_log > 1) PRINT(log);
  }

  void SampleRenderer::createPipeline() {
    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : rayGenPrograms)
      programGroups.push_back(pg);
    for (auto pg : missPrograms)
      programGroups.push_back(pg);
    for (auto pg : hitgroupPrograms)
      programGroups.push_back(pg);

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixPipelineCreate(optixContext,
                                    &pipelineCompileOptions,
                                    &pipelineLinkOptions,
                                    programGroups.data(),
                                    (int)programGroups.size(),
                                    log,&sizeof_log,
                                    &pipeline
                                    ));
    if (sizeof_log > 1) PRINT(log);

    OPTIX_CHECK(optixPipelineSetStackSize
                (/* [in] The pipeline to configure the stack size for */
                 pipeline,
                 /* [in] The direct stack size requirement for direct
                    callables invoked from IS or AH. */
                 2*1024,
                 /* [in] The direct stack size requirement for direct
                    callables invoked from RG, MS, or CH.  */
                 2*1024,
                 /* [in] The continuation stack requirement. */
                 2*1024,
                 /* [in] The maximum depth of a traversable graph
                    passed to trace. */
                 1));
    if (sizeof_log > 1) PRINT(log);
  }

  void SampleRenderer::buildSBT() {
    // ------------------------------------------------------------------
    // build raygen records
    // ------------------------------------------------------------------
    std::vector<RaygenRecord> raygenRecords;
    for (int i=0;i<rayGenPrograms.size();i++) {
      RaygenRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(rayGenPrograms[i],&rec));
      rec.data = nullptr; /* for now ... */
      raygenRecords.push_back(rec);
    }
    raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<MissRecord> missRecords;
    for (int i=0;i<missPrograms.size();i++) {
      MissRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(missPrograms[i],&rec));
      rec.data = nullptr; /* for now ... */
      missRecords.push_back(rec);
    }
    missRecordsBuffer.alloc_and_upload(missRecords);
    sbt.missRecordBase          = missRecordsBuffer.d_pointer();
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount         = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------

    // we don't actually have any objects in this example, but let's
    // create a dummy one so the SBT doesn't have any null pointers
    // (which the sanity checks in compilation would complain about)
    int numObjects = 1;
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int i=0;i<numObjects;i++) {
      int objectType = 0;
      HitgroupRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPrograms[objectType],&rec));
      rec.objectID = i;
      hitgroupRecords.push_back(rec);
    }
    hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount         = (int)hitgroupRecords.size();
  }

  void SampleRenderer::render() {
    // sanity check: make sure we launch only after first resize is
    // already done:
    if (launchParams.fbSize.x == 0) return;
    launchParamsBuffer.upload(&launchParams,1);
    launchParams.frameID++;

    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                            pipeline,cuStream,
                            /*! parameters and SBT */
                            launchParamsBuffer.d_pointer(),
                            launchParamsBuffer.sizeInBytes,
                            &sbt,
                            /*! dimensions of the launch: */
                            launchParams.fbSize.x,
                            launchParams.fbSize.y,
                            1
                            ));
    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    CUDA_SYNC_CHECK();
  }

  /*! resize frame buffer to given resolution */
  void SampleRenderer::resize(const vec2i &newSize)
  {
    // if window minimized
    if (newSize.x == 0 | newSize.y == 0) return;

    // resize our cuda frame buffer
    colorBuffer.resize(newSize.x*newSize.y*sizeof(uint32_t));
    // update the launch parameters that we'll pass to the optix
    // launch:
    launchParams.fbSize      = newSize;
    launchParams.colorBuffer = (uint32_t*)colorBuffer.d_ptr;
  }

  /*! download the rendered color buffer */
  void SampleRenderer::downloadPixels(uint32_t h_pixels[])
  {
    colorBuffer.download(h_pixels,
                         launchParams.fbSize.x*launchParams.fbSize.y);
  }


}
