cmake_minimum_required(VERSION 3.18)
project(CUDA_Project LANGUAGES CUDA CXX)

# Set CUDA architecture (adjust as needed)
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xptxas -dlcm=ca --use_fast_math -lineinfo")

# Enable CUDA compilation with NVCC
enable_language(CUDA)

# Optionally, find the CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Define source and include directories
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
include_directories(${INCLUDE_DIR})

# Create executable for comparison.cu
add_executable(comparison ${SRC_DIR}/comparison.cu)
set_target_properties(comparison PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(comparison PRIVATE CUDA::cudart)

# Create executable for table.cu
add_executable(table ${SRC_DIR}/table.cu)
set_target_properties(table PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(table PRIVATE CUDA::cudart)
