#include "convolutional_ops.h"

#include <cuda_fp16.h>
#include <ATen/cuda/detail/KernelUtils.h>


namespace
{
#if defined(USE_ROCM) || defined(TORCH_USE_HIP_DSA)
    __device__ inline __half operator+(const __half &a, const __half &b)
    {
        return __hadd(a, b);
    }

    __device__ inline __half operator-(const __half &a, const __half &b)
    {
        return __hsub(a, b);
    }

    __device__ inline __half operator*(const __half &a, const __half &b)
    {
        return __hmul(a, b);
    }

    __device__ inline __half operator/(const __half &a, const __half &b)
    {
        return __hdiv(a, b);
    }

    __device__ inline __half &operator+=(__half &a, const __half &b)
    {
        a = __hadd(a, b);
        return a;
    }
#endif
}

template <typename scalar_t>
__global__ void apply_kernel_maps_to_irradiance_maps
(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input_irradiance,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> kernel_maps,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output,
    const int B,
    const int C,
    const int C_km,
    const int H,
    const int W
)
{
    const int b = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int kernel_center_h = c / W;
    const int kernel_center_w = c % W;

    if (b >= B || kernel_center_h >= H || kernel_center_w >= W) return;

    int count = 0;
    const int kernel_half_width = 3;
    scalar_t weight_sum = 0;
    scalar_t weighted_color_sum[3];

    weighted_color_sum[0] = 0;
    weighted_color_sum[1] = 0;
    weighted_color_sum[2] = 0;

    for (int i = -kernel_half_width; i <= kernel_half_width; ++i)
    {
        for (int j = -kernel_half_width; j <= kernel_half_width; ++j)
        {
            auto h = max(0, min(kernel_center_h + i, H - 1));
            auto w = max(0, min(kernel_center_w + j, W - 1));

            // int flat_idx_km = b * C_km * H * W + count * H * W + kernel_center_h * W + kernel_center_w;
            // scalar_t weight = kernel_maps[flat_idx_km];
            scalar_t weight = kernel_maps[b][count][kernel_center_h][kernel_center_w];
            weight_sum += weight;

            for (int k = 0; k < 3; ++k)
            {
                // int flat_idx = b * C * H * W + k * H * W + h * W + w;
                // scalar_t irradiance_k = input_irradiance[flat_idx];
                scalar_t irradiance_k = input_irradiance[b][k][h][w];

                weighted_color_sum[k] += weight * irradiance_k;
            }

            count ++;
        }
    }
    
    for (int i = 0; i < 3; ++i)
    {
        // int flat_idx = b * C * H * W + i * H * W + kernel_center_h * W + kernel_center_w;
        // output[flat_idx] = weighted_color_sum[i] / (weight_sum + (scalar_t)(0.0001f));
        output[b][i][kernel_center_h][kernel_center_w] = weighted_color_sum[i] / (weight_sum + (scalar_t)(0.0001f));
    }
}

void apply_kernel_maps_to_irradiance_maps_launcher
(
    torch::Tensor input_irradiance,
    torch::Tensor kernel_maps,
    torch::Tensor output
)
{
    const auto batch_size = input_irradiance.size(0);
    const auto channels = input_irradiance.size(1);
    const auto channels_km = kernel_maps.size(1);
    const auto height = input_irradiance.size(2);
    const auto width = input_irradiance.size(3);
    const auto type = input_irradiance.scalar_type();

    const int total_threads = batch_size * height * width;
    const int num_threads_per_block = 1024;
    auto num_blocks = at::cuda::detail::GET_BLOCKS(total_threads, num_threads_per_block);

#if defined(USE_ROCM) || defined(TORCH_USE_HIP_DSA)
    const hipStream_t stream = at::cuda::getCurrentHIPStream();
#else
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#endif

    AT_DISPATCH_FLOATING_TYPES_AND_HALF
    (
        type, 
        "apply_kernel_maps_to_irradiance_maps",
        ([&] {
		    apply_kernel_maps_to_irradiance_maps<scalar_t><<<num_blocks, num_threads_per_block, 0, stream>>>
		    (
            	input_irradiance.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
		    	kernel_maps.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
		    	output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                batch_size,
                channels,
                channels_km,
                height,
                width
		    );
        })
    );
}