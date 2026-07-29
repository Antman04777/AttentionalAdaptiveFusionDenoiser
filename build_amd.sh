hipify-perl bilateral_grid_kernels.cu > bilateral_grid_kernels.hip
rm -rf build
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
make -j$(nproc)
