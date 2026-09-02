rm -rf build_release
mkdir build_release
cd build_release
cmake -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DARCH_IS_CUDA=ON -DNDEBUG=ON ..
make -j$(nproc)