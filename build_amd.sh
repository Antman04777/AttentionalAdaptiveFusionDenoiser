hipify-perl convolutional_ops.cu > convolutional_ops.hip
rm -rf build
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DARCH_IS_CUDA=OFF ..
make -j$(nproc)
