cmake_minimum_required(VERSION 3.17)

project(PathTracing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OptiX_INCLUDE "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0/include")
set(CUDA_INCLUDE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include")
set(Optix_LIBRARY "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0/SDK/lib")
set(CUDA_LIBRARIES "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64")
set(CUDA_CUDA_LIBRARY "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0/SDK/cuda")

include_directories(
        ${OptiX_INCLUDE}
        ${CUDA_INCLUDE}
)

add_executable(PathTracing
  optix7.h
  main.cpp
)


find_package(CUDAToolkit REQUIRED)

# 添加 CUDA 编译器配置
set_target_properties(PathTracing PROPERTIES
        CUDA_ARCHITECTURES "64"           # 设置目标机型为 64 位
        CUDA_RUNTIME_LIBRARY Static      # 使用静态 CUDA 运行时
)

target_link_directories(PathTracing PRIVATE
  gdt
 ${Optix_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
)

# 链接库
target_link_libraries(PathTracing PRIVATE
        opengl32
        cuda
        cudadevrt
        cudart
        cudart_static
        cufft
        cufftw
        curand
        cusolver
        cusparse
        nvblas
)

# NVCC 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptx") # 输出 ptx 文件