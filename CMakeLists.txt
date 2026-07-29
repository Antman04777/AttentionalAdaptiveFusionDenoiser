cmake_minimum_required(VERSION 3.18)
project(bilateral_grid_kernels  LANGUAGES CXX HIP)

set(CMAKE_HIP_COMPILER "/opt/rocm/bin/hipcc")

find_package(Torch REQUIRED)

add_library(bilateral_grid_kernels SHARED
    bilateral_grid_kernels.hip
    bilateral_grid_kernels.cpp
)

set_target_properties(bilateral_grid_kernels PROPERTIES
    HIP_ARCHITECTURES "${PYTORCH_ROCM_ARCH}"
)

target_link_libraries(bilateral_grid_kernels PRIVATE ${TORCH_LIBRARIES})
