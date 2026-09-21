#include <cuda_fp16.h>
#include <ATen/cuda/detail/KernelUtils.h>

namespace
{
    /*
    __device__ inline float3 operator*(const float &a, const float3 &b)
    {
        return make_float3(a * b.x, a * b.y, a * b.z);
    }

    __device__ inline float3 operator*(const float3 &a, const float3 &b)
    {
        return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
    }

    __device__ inline float3 operator+(const float3 &a, const float3 &b)
    {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    __device__ inline float3& operator+=(float3& a, float3& b)
    {
        a.x += b.x;
        a.y += b.y;
        a.z += b.z;
        return a;
    }
    */
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

    __device__ inline __half operator

}

__global__ void apply_kernel_maps_to_irradiance_maps
(
    const torch::PackedTensorAccessor<__half, 4, torch::RestrictPtrTraits, size_t> input_irradiance,
    const torch::PackedTensorAccessor<__half, 4, torch::RestrictPtrTraits, size_t> albedo,
    const torch::PackedTensorAccessor<__half, 4, torch::RestrictPtrTraits, size_t> kernel_maps,
    torch::PackedTensorAccessor<__half, 4, torch::RestrictPtrTraits, size_t> output
)
{
    const int B = input_irradiance.size(0)
    const int H = input_irradiance.size(2)
    const int W = input_irradiance.size(3)

    const int b = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int kernel_center_h = c / W;
    const int kernel_center_w = c % W;

    if (b >= B || idx_h >= H || idx_w >= W) return;

    const int count = 0;
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
            h = max(0, min(kernel_center_h + i, H - 1));
            w = max(0, min(kernel_center_w + j, W - 1));

            weight = kernel_maps[b][count][kernel_center_h][kernel_center_w];
            weight_sum += weight;

            __half irradiance_r = input_irradiance[b][0][h][w];
            __half irradiance_g = input_irradiance[b][1][h][w];
            __half irradiance_b = input_irradiance[b][2][h][w];

            weighted_color_sum[0] += weight * irradiance_r;
            weighted_color_sum[1] += weight * irradiance_g;
            weighted_color_sum[2] += weight * irradiance_b;

            count ++;
        }
    }
    
    for (int i = 0; i < 3; ++i)
    {
        output[b][i][kernel_center_h][kernel_center_w] = (weighted_color_sum[i] / weight_sum) * albedo[b][i][kernel_center_h][kernel_center_w];
    }
}

void apply_kernel_maps_to_irradiance_maps_launcher
(
    torch::Tensor input_irradiance,
    torch::Tensor albedo,
    torch::Tensor kernel_maps,
    torch::Tensor output
)
{
    const auto batch_size = input_irradiance.input(0);
    const auto height = input_irradiance.input(2);
    const auto width = input_irradiance.input(3);
    const auto type = input_irradiance.scalar_type();

    const int total_threads = batch_size * height * width;
    const int num_threads_per_block = 1024;
    auto num_blocks = at::cuda::detail::GET_BLOCKS(total_threads, num_threads_per_block);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF
    (
        type, 
        "apply_kernel_maps_to_irradiance_maps",
        ([&] {
		    apply_kernel_maps_to_irradiance_maps<<<num_blocks, num_threads_per_block>>>
		    (
            	input_irradiancepacked_accessor<__half, 4, torch::RestrictPtrTraits, size_t>(),
		    	albedo.packed_accessor<__half, 4, torch::RestrictPtrTraits, size_t>(),
		    	kernel_maps.packed_accessor<__half, 4, torch::RestrictPtrTraits, size_t>(),
		    	output.packed_accessor<__half, 4, torch::RestrictPtrTraits, size_t>()
		    );
        })
    );
}