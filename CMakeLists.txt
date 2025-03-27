cmake_minimum_required(VERSION 3.17)

project(PathTracing LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Optix_Root "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0/")
set(OptiX_INCLUDE "${Optix_Root}/include")
set(CMAKE_MODULE_PATH "${Optix_Root}/SDK/CMake")
#set(Optix_LIBRARY "${Optix_Root}/SDK/build/bin/Debug")

set(Cuda_Root "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
set(CUDA_INCLUDE "${Cuda_Root}/include")
set(CUDA_LIBRARIES "${Cuda_Root}/lib/x64")

find_package(CUDA REQUIRED)
find_package(OptiX REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_program(BIN2C bin2c DOC "Path to the cuda-sdk bin2c executable.")

# this macro defines cmake rules that execute the following four steps:
# 1) compile the given cuda file ${cuda_file} to an intermediary PTX file
# 2) use the 'bin2c' tool (that comes with CUDA) to
#    create a second intermediary (.c-)file which defines a const string variable
#    (named '${c_var_name}') whose (constant) value is the PTX output
#    from the previous step.
# 3) compile the given .c file to an intermediary object file (why thus has
#    that PTX string 'embedded' as a global constant.
# 4) assign the name of the intermediary .o file to the cmake variable
#    'output_var', which can then be added to cmake targets.
macro(cuda_compile_and_embed output_var cuda_file)
  set(c_var_name ${output_var})
  cuda_compile_ptx(ptx_files ${cuda_file})
  list(GET ptx_files 0 ptx_file)
  set(embedded_file ${ptx_file}_embedded.c)
  message(STATUS "adding rule to compile and embed ${cuda_file} to \"const char ${c_var_name}[] in file ${embedded_file};\"")
  add_custom_command(
    OUTPUT ${embedded_file}
    COMMAND ${BIN2C} -c --padd 0 --type char --name ${c_var_name} ${ptx_file} > ${embedded_file}
    DEPENDS ${ptx_file}
    COMMENT "compiling (and embedding ptx from) ${cuda_file}"
    )
  set(${output_var} ${embedded_file})
  message(STATUS "${output_var}= ${${output_var}}")
endmacro()

include_directories(
        ${OptiX_INCLUDE}
        ${CUDA_INCLUDE}
        ${CMAKE_CURRENT_SOURCE_DIR}
)

file(GLOB IMGUI_SOURCES
    3rdParty/imgui/*.cpp
    3rdParty/imgui/backends/imgui_impl_glfw.cpp
    3rdParty/imgui/backends/imgui_impl_opengl3.cpp
)

add_library(imgui STATIC ${IMGUI_SOURCES})
target_include_directories(imgui PUBLIC
    3rdParty/imgui
    3rdParty/imgui/backends
    3rdParty/glfw/include
)

cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)

add_executable(PathTracing
    ${embedded_ptx_code}
    optix7.h
    CUDABuffer.h
    LaunchParams.h
    SampleRenderer.h
    SampleRenderer.cpp
    main.cpp
    glfWindow/GLFWindow.cpp
    Model.h
    Model.cpp
        Interaction.h
        PostProcess.h
)

# 添加 CUDA 编译器配置
set_target_properties(PathTracing PROPERTIES
        CUDA_ARCHITECTURES "61"           # 设置目标机型为 GTX 10
        CUDA_RUNTIME_LIBRARY Static      # 使用静态 CUDA 运行时
)

target_link_directories(PathTracing PRIVATE
  ${CUDA_LIBRARIES}
)

add_subdirectory(3rdParty/glfw)

# 链接库
target_link_libraries(PathTracing PRIVATE
        opengl32
        glfw
        cuda
        #cudadevrt
        cudart
        #cudart_static
        #cufft
        #cufftw
        #curand
        #cusolver
        #cusparse
        #nvblas
        imgui
)
