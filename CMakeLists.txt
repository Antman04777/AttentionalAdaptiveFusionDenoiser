cmake_minimum_required(VERSION 3.18)
if(ARCH_IS_CUDA)
    project(convolutional_ops LANGUAGES CXX CUDA)
else()
    project(convolutional_ops LANGUAGES CXX HIP)
endif()

if(NOT ARCH_IS_CUDA)
    set(CMAKE_HIP_COMPILER "/opt/rocm/bin/hipcc")
endif()

find_package(Torch REQUIRED)

if(ARCH_IS_CUDA)
    add_library(convolutional_ops SHARED
    convolutional_ops.cu
    convolutional_ops.cpp)
else()
    add_library(convolutional_ops SHARED
    convolutional_ops.hip
    convolutional_ops.cpp)
endif()

if(ARCH_IS_CUDA)
    set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")
    set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
else()
    set_target_properties(convolutional_ops PROPERTIES
        HIP_ARCHITECTURES "${PYTORCH_ROCM_ARCH}"
    )
endif()

if(ARCH_IS_CUDA)
    target_compile_options(convolutional_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    target_compile_options(convolutional_ops PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
else()
    target_compile_options(convolutional_ops PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3>)
    target_compile_options(convolutional_ops PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
endif()

target_link_libraries(convolutional_ops PRIVATE ${TORCH_LIBRARIES})