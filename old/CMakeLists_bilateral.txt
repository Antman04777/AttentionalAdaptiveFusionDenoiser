cmake_minimum_required(VERSION 3.18)
if(ARCH_IS_CUDA)
    project(bilateral_grid_kernels LANGUAGES CXX CUDA)
else()
    project(bilateral_grid_kernels  LANGUAGES CXX HIP)
endif()

if(NOT ARCH_IS_CUDA)
    set(CMAKE_HIP_COMPILER "/opt/rocm/bin/hipcc")
endif()

find_package(Torch REQUIRED)

if(ARCH_IS_CUDA)
    add_library(bilateral_grid_kernels SHARED
    bilateral_grid_kernels.cu
    bilateral_grid_kernels.cpp)
else()
    add_library(bilateral_grid_kernels SHARED
    bilateral_grid_kernels.hip
    bilateral_grid_kernels.cpp)
endif()

if(ARCH_IS_CUDA)
    set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")
    set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
else()
    set_target_properties(bilateral_grid_kernels PROPERTIES
        HIP_ARCHITECTURES "${PYTORCH_ROCM_ARCH}"
    )
endif()

if(ARCH_IS_CUDA)
    target_compile_options(bilateral_grid_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    target_compile_options(bilateral_grid_kernels PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
else()
    target_compile_options(bilateral_grid_kernels PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3>)
    target_compile_options(bilateral_grid_kernels PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
endif()

target_link_libraries(bilateral_grid_kernels PRIVATE ${TORCH_LIBRARIES})
