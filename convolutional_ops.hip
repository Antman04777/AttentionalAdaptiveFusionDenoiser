#include "hip/hip_runtime.h"
#include "convolutional_ops.h"

#include <hip/hip_fp16.h>
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

// __global__ void apply_kernel_maps_to_irradiance_maps
// (
//     const torch::PackedTensorAccessor32<__half, 4> input_irradiance,
//     const torch::PackedTensorAccessor32<__half, 4> kernel_maps,
//     torch::PackedTensorAccessor32<__half, 4> output
// )
// {
//     const int B = input_irradiance.size(0);
//     const int H = input_irradiance.size(2);
//     const int W = input_irradiance.size(3);

//     const int b = blockIdx.y;
//     const int c = blockIdx.x * blockDim.x + threadIdx.x;
//     const int kernel_center_h = c / W;
//     const int kernel_center_w = c % W;

//     if (b >= B || kernel_center_h >= H || kernel_center_w >= W) return;

//     int count = 0;
//     const int kernel_half_width = 3;
//     __half weight_sum = __float2half(0.0f);
//     __half weighted_color_sum[3];

//     weighted_color_sum[0] = __float2half(0.0f);
//     weighted_color_sum[1] = __float2half(0.0f);
//     weighted_color_sum[2] = __float2half(0.0f);

//     for (int i = -kernel_half_width; i <= kernel_half_width; ++i)
//     {
//         for (int j = -kernel_half_width; j <= kernel_half_width; ++j)
//         {
//             auto h = max(0, min(kernel_center_h + i, H - 1));
//             auto w = max(0, min(kernel_center_w + j, W - 1));

//             __half weight = kernel_maps[b][count][kernel_center_h][kernel_center_w];
//             weight_sum += weight;

//             __half irradiance_r = input_irradiance[b][0][h][w];
//             __half irradiance_g = input_irradiance[b][1][h][w];
//             __half irradiance_b = input_irradiance[b][2][h][w];

//             weighted_color_sum[0] += weight * irradiance_r;
//             weighted_color_sum[1] += weight * irradiance_g;
//             weighted_color_sum[2] += weight * irradiance_b;

//             count ++;
//         }
//     }
    
//     for (int i = 0; i < 3; ++i)
//     {
//         output[b][i][kernel_center_h][kernel_center_w] = (weighted_color_sum[i] / weight_sum);
//     }
// }

__global__ void apply_kernel_maps_to_irradiance_maps
(
    const __half *input_irradiance,
    const __half *kernel_maps,
    __half *output,
    const int B,
    const int C,
    const int C_km,
    const int H,
    const int W
)
{
    // const int B = input_irradiance.size(0);
    // const int C = input_irradiance.size(1);
    // const int H = input_irradiance.size(2);
    // const int W = input_irradiance.size(3);

    // const int C_km = 

    const int b = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int kernel_center_h = c / W;
    const int kernel_center_w = c % W;

    if (b >= B || kernel_center_h >= H || kernel_center_w >= W) return;

    int count = 0;
    const int kernel_half_width = 3;
    __half weight_sum = __float2half(0.0f);
    __half weighted_color_sum[3];

    weighted_color_sum[0] = __float2half(0.0f);
    weighted_color_sum[1] = __float2half(0.0f);
    weighted_color_sum[2] = __float2half(0.0f);

    for (int i = -kernel_half_width; i <= kernel_half_width; ++i)
    {
        for (int j = -kernel_half_width; j <= kernel_half_width; ++j)
        {
            auto h = max(0, min(kernel_center_h + i, H - 1));
            auto w = max(0, min(kernel_center_w + j, W - 1));

            // __half weight = kernel_maps[b][count][kernel_center_h][kernel_center_w];
            int flat_idx_km = b * C_km * H * W + count * H * W + kernel_center_h * W + kernel_center_w;
            __half weight = kernel_maps[flat_idx_km];
            weight_sum += weight;

            for (int k = 0; k < 3; ++k)
            {
                int flat_idx = b * C * H * W + k * H * W + h * W + w;
                __half irradiance_k = input_irradiance[flat_idx];

                weighted_color_sum[k] += weight * irradiance_k;
            }

            count ++;
        }
    }
    
    for (int i = 0; i < 3; ++i)
    {
        int flat_idx = b * C * H * W + i * H * W + kernel_center_h * W + kernel_center_w;
        // output[b][i][kernel_center_h][kernel_center_w] = (weighted_color_sum[i] / weight_sum);
        output[flat_idx] = weighted_color_sum[i] / (weight_sum + __float2half(0.0001f));
    }
}

void apply_kernel_maps_to_irradiance_maps_launcher
(
    // torch::Tensor input_irradiance,
    // torch::Tensor kernel_maps,
    // torch::Tensor output
    const at::Half *input_irradiance,
	const at::Half *kernel_maps,
	at::Half *output,
    const int batch_size,
    const int channels,
    const int channels_km,
    const int height,
    const int width,
    const torch::Dtype type
)
{
    // const auto batch_size = input_irradiance.size(0);
    // const auto channels = input_irradiance.size(1);
    // const auto channels_km = kernel_maps.size(1);
    // const auto height = input_irradiance.size(2);
    // const auto width = input_irradiance.size(3);
    // const auto type = input_irradiance.scalar_type();

    const int total_threads = batch_size * height * width;
    const int num_threads_per_block = 1024;
    auto num_blocks = at::cuda::detail::GET_BLOCKS(total_threads, num_threads_per_block);

    // const auto input_irradiance_data_ptr = input_irradiance.const_data_ptr<at::Half>();
    // const auto kernel_maps_data_ptr = kernel_maps.const_data_ptr<at::Half>();
    // auto output_data_ptr = input_irradiance.data_ptr<at::Half>();

    const __half *__hinput_irradiance = reinterpret_cast<const __half *>(input_irradiance);
	const __half *__hkernel_maps = reinterpret_cast<const __half *>(kernel_maps);
	__half *__houtput = reinterpret_cast<__half *>(output);

#if defined(USE_ROCM) || defined(TORCH_USE_HIP_DSA)
    const hipStream_t stream = at::cuda::getCurrentHIPStream();
#else
    const hipStream_t stream = at::cuda::getCurrentCUDAStream();
#endif

    AT_DISPATCH_FLOATING_TYPES_AND_HALF
    (
        type, 
        "apply_kernel_maps_to_irradiance_maps",
        ([&] {
		    apply_kernel_maps_to_irradiance_maps<<<num_blocks, num_threads_per_block, 0, stream>>>
		    (
            	__hinput_irradiance,
		    	__hkernel_maps,
		    	__houtput,
                batch_size,
                channels,
                channels_km,
                height,
                width
		    );
        })
    );
}