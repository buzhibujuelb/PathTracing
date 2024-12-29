cmake_minimum_required(VERSION 3.17)

project(PathTracing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Optix_Root "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0/")
set(OptiX_INCLUDE "${Optix_Root}/include")
#set(Optix_LIBRARY "${Optix_Root}/SDK/build/bin/Debug")

set(Cuda_Root "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
set(CUDA_INCLUDE "${Cuda_Root}/include")
set(CUDA_LIBRARIES "${Cuda_Root}/lib/x64")

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
  ${CUDA_LIBRARIES}
)

# 链接库
target_link_libraries(PathTracing PRIVATE
        #opengl32
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
)

# NVCC 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptx") # 输出 ptx 文件