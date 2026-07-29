#include "hip/hip_runtime.h"
#include "bilateral_grid_kernels.h"
#include <ATen/cuda/detail/KernelUtils.h>

__device__ float diff_abs(float x) {
  float eps = 1e-8;
  return sqrt(x * x + eps);
}

__device__ float d_diff_abs(float x) {
  float eps = 1e-8;
  return x / sqrt(x * x + eps);
}

__device__ float weight_z(float x) {
  float abx = diff_abs(x);
  return max(1.0f - abx, 0.0f);
}

__device__ float d_weight_z(float x) {
  float abx = diff_abs(x);
  if (abx > 1.0f) {
    return 0.0f;
    // return abx;
  } else {
    return d_diff_abs(x);
  }
}

__global__ void CreateBiGridKernel(
    int num_grid_points, const float *input_image, const float *input_guide,
    const int batch_size, const int image_height, const int image_width,
    const int channel_size, const int grid_height, const int grid_width,
    const int grid_depth, const int sr, float *out_color) {

  CUDA_KERNEL_LOOP(i, num_grid_points) {
    int ss = image_width / grid_width;

    int c = i % channel_size;
    int w = (i / channel_size) % grid_width;
    int h = (i / (channel_size * grid_width)) % grid_height;
    int b = (i / (channel_size * grid_width * grid_height)) % batch_size;

    float f_w = w + 0.5f;
    float f_h = h + 0.5f;

    int left = static_cast<int>(ceil((w - 0.5f) * ss));
    int right = left + 2 * ss;
    int up = static_cast<int>(ceil((h - 0.5f) * ss));
    int down = up + 2 * ss;

    int sx = channel_size;
    int sy = channel_size * image_width;
    int sb = channel_size * image_width * image_height;

    float sum_color[64];
    float sum_count[64];
    for (int dd = 0; dd < grid_depth; dd++) {
      sum_color[dd] = 0.0f;
      sum_count[dd] = 0.0f;
    }

    for (int yy = up; yy < down; yy++) {
      if (yy < 0 || yy > image_height - 1)
        continue;
      float wy = weight_z((yy + 0.5f) / ss - f_h);
      for (int xx = left; xx < right; xx++) {
        if (xx < 0 || xx > image_width - 1)
          continue;
        float wx = weight_z((xx + 0.5f) / ss - f_w);

        int idx_guide = xx + yy * image_width + b * image_width * image_height;
        int idx_image = c + xx * sx + yy * sy + b * sb;

        float guide_val = (input_guide[idx_guide] + 0.5f) / sr;
        int d_lower = static_cast<int>(max(ceil(guide_val - 1.5f), 0.0f));
        int d_upper =
            static_cast<int>(min(floor(guide_val + 0.5f), grid_depth - 1.0f));
        for (int zz = d_lower; zz <= d_upper; zz++) {
          float wz = weight_z(guide_val - zz - 0.5f);
          float wt = wy * wx * wz;
          sum_color[zz] += input_image[idx_image] * wt;
          sum_count[zz] += wt;
        }
      }
    }
    int idx_base = w * channel_size * grid_depth +
                   h * channel_size * grid_depth * grid_width +
                   b * channel_size * grid_depth * grid_width * grid_height;
    for (int dd = 0; dd < grid_depth; dd++) {
      int idx_grid = c + dd * sx + idx_base;
      out_color[idx_grid] =
          abs(sum_count[dd]) > 0 ? sum_color[dd] / sum_count[dd] : 0.0f;
    }
  }
}

void create_grid_kernel_launcher(const float *input_image,
                                 const float *input_guide, const int batch_size,
                                 const int image_height, const int image_width,
                                 const int channel_size, const int grid_height,
                                 const int grid_width, const int grid_depth,
                                 const int sr, float *const out_color,
                                 torch::Device &device,
                                 torch::Dtype &type)
                                 {
  int64_t num_grid_points =
      batch_size * grid_height * grid_width * channel_size;
  torch::Device current_device(torch::kCUDA, 0);

  if (current_device == device && num_grid_points > 0) {
    int64_t num_blocks = at::cuda::detail::GET_BLOCKS(num_grid_points);

#if defined(USE_ROCM) || defined(TORCH_USE_HIP_DSA)
    const hipStream_t stream = at::cuda::getCurrentHIPStream();
#else
    const hipStream_t stream = at::cuda::getCurrentCUDAStream();
#endif

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        type, "CreateBiGridKernel", ([&] {
          CreateBiGridKernel<<<num_blocks, at::cuda::detail::CUDA_NUM_THREADS, 0, stream>>>(
                  num_grid_points, input_image, input_guide, batch_size,
                  image_height, image_width, channel_size, grid_height,
                  grid_width, grid_depth, sr, out_color);
        }));
  }
}
