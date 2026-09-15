#include "bilateral_grid_kernels.h"
#include <ATen/cuda/detail/KernelUtils.h>

#ifdef TRAINING
__device__ float diff_abs(float x)
{
	float eps = 1e-8;
	return sqrt(x * x + eps);
}

__device__ float d_diff_abs(float x)
{
	float eps = 1e-8;
	return x / sqrt(x * x + eps);
}

__device__ float weight_z(float x)
{
	float abx = diff_abs(x);
	return max(1.0f - abx, 0.0f);
}

__device__ float d_weight_z(float x)
{
	float abx = diff_abs(x);
	if (abx > 1.0f) {
		return 0.0f;
		// return abx;
	} else {
		return d_diff_abs(x);
	}
}
#else
// #include <cuda_fp16.h>

__device__ float diff_abs(float x)
{
	float eps = 1e-8;
	return sqrt(x * x + eps);
}

__device__ float d_diff_abs(float x)
{
	float eps = 1e-8;
	return x / sqrt(x * x + eps);
}

__device__ float weight_z(float x)
{
	float abx = diff_abs(x);
	return max(1.0f - abx, 0.0f);
}

__device__ float d_weight_z(float x)
{
	float abx = diff_abs(x);
	if (abx > 1.0f) {
		return 0.0f;
		// return abx;
	} else {
		return d_diff_abs(x);
	}
}

// overloaded for half
__device__ __half diff_abs(__half x)
{
	__half eps = __float2half(static_cast<float>(1e-8));
	return hsqrt(__hadd(__hmul(x, x), eps));
}

__device__ __half d_diff_abs(__half x)
{
	__half eps = __float2half(static_cast<float>(1e-8));
	return __hdiv(x, hsqrt(__hadd(__hmul(x, x), eps)));
}

__device__ __half weight_z(__half x)
{
	__half abx = diff_abs(x);
	return __hmax(__hsub(__float2half(1.0f), abx), __float2half(0.0f));
}

__device__ __half d_weight_z(__half x)
{
	__half abx = diff_abs(x);
	if (__hgt(abx, __float2half(1.0f))) {
		return __float2half(0.0f);
		// return abx;
	} else {
		return d_diff_abs(x);
	}
}
#endif

#ifdef TRAINING
__global__ void CreateBiGridKernel(
    int num_grid_points,
	const float *input_image,
	const float *input_guide,
    const int batch_size,
	const int image_height,
	const int image_width,
    const int channel_size,
	const int grid_height,
	const int grid_width,
    const int grid_depth,
	const int sr,
	float *out_color
)
{
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
          abs(sum_count[dd]) > 0 ? sum_color[dd] / sum_count[dd] : 0.0f; // sum count is 0 for some reason
    }
  }
}
#else
__global__ void CreateBiGridKernel(
    int num_grid_points,
	const __half *input_image,
	const __half *input_guide,
    const int batch_size,
	const int image_height,
	const int image_width,
    const int channel_size,
	const int grid_height,
	const int grid_width,
    const int grid_depth,
	const int sr,
	__half *out_color
) 

{
  CUDA_KERNEL_LOOP(i, num_grid_points) {
    int ss = image_width / grid_width;

    int c = i % channel_size;
    int w = (i / channel_size) % grid_width;
    int h = (i / (channel_size * grid_width)) % grid_height;
    int b = (i / (channel_size * grid_width * grid_height)) % batch_size;

    __half f_w = __hadd(__float2half(static_cast<float>(w)), __float2half(0.5f));
    __half f_h = __hadd(__float2half(static_cast<float>(h)), __float2half(0.5f));

    int left = static_cast<int>(ceil((w - 0.5) * ss));
    int right = left + 2 * ss;
    int up = static_cast<int>(ceil((h - 0.5) * ss));
    int down = up + 2 * ss;

    int sx = channel_size;
    int sy = channel_size * image_width;
    int sb = channel_size * image_width * image_height;

    __half sum_color[64];
    __half sum_count[64];
    for (int dd = 0; dd < grid_depth; dd++) {
    	sum_color[dd] = __float2half(0.0f);
    	sum_count[dd] = __float2half(0.0f);
    }

    for (int yy = up; yy < down; yy++) {
      	if (yy < 0 || yy > image_height - 1)
      		continue;
		// float wy = weight_z((yy + 0.5f) / ss - f_h);
      	__half wy = weight_z(__hsub(__hdiv(__hadd(__float2half(static_cast<float>(yy)), __float2half(0.5f)), __float2half(static_cast<float>(ss))), f_h));
      	for (int xx = left; xx < right; xx++) {
      	  	if (xx < 0 || xx > image_width - 1)
      	    	continue;
			// float wx = weight_z((xx + 0.5f) / ss - f_w);
      	  	__half wx = weight_z(__hsub(__hdiv(__hadd(__float2half(static_cast<float>(xx)), __float2half(0.5f)), __float2half(static_cast<float>(ss))), f_w));

      	  	int idx_guide = xx + yy * image_width + b * image_width * image_height;
      	  	int idx_image = c + xx * sx + yy * sy + b * sb;

      	  	__half guide_val = __hdiv((__hadd(input_guide[idx_guide], __float2half(0.5f))), __float2half(static_cast<float>(sr)));
      	  	// int d_lower = static_cast<int>(max(ceil(guide_val - 1.5), 0.0));
			int d_lower = static_cast<int>(__half2float(__hmax(hceil(__hsub(guide_val, __float2half(1.5f))), __float2half(0.0f))));
      	  	int d_upper = static_cast<int>(__half2float(__hmin(hfloor(__hadd(guide_val, __float2half(0.5f))), __hsub(__float2half(static_cast<float>(grid_depth)), __float2half(1.0f)))));

      	  	for (int zz = d_lower; zz <= d_upper; zz++) {
      	    	__half wz = weight_z(__hsub(__hsub(guide_val, __float2half(static_cast<float>(zz))), __float2half(0.5f)));
      	    	__half wt = __hmul(__hmul(wy, wx), wz);
      	    	sum_color[zz] = __hadd(sum_color[zz], __hmul(input_image[idx_image], wt));
      	    	sum_count[zz] = __hadd(sum_count[zz], wt);
      	  	}
      	}
    }
    int idx_base = w * channel_size * grid_depth +
                   h * channel_size * grid_depth * grid_width +
                   b * channel_size * grid_depth * grid_width * grid_height;

    for (int dd = 0; dd < grid_depth; dd++) {
    	int idx_grid = c + dd * sx + idx_base;
    	out_color[idx_grid] = __hgt(__habs(sum_count[dd]), __float2half(0.0f)) ? __hdiv(sum_color[dd], sum_count[dd]) : __float2half(0.0f);
    }
  }
}
#endif

__global__ void CreateBiGridImageGradKernel(
	int num_image_pixels,
	const float* input_image, 
	const float* input_guide,
	const float* backprop,
	const int batch_size,
	const int image_height,
	const int image_width,
	const int channel_size,
	const int grid_height, 
	const int grid_width, 
	const int grid_depth,
	const int sr,
	float* image_grad
) {
	CUDA_KERNEL_LOOP(i, num_image_pixels)
	{
		int ss = image_width / grid_width;

		int c = i % channel_size;
		int im_w = (i / channel_size) % image_width;
		int im_h = (i / (channel_size * image_width)) % image_height;
		int b = (i / (channel_size * image_width * image_height)) % batch_size;
		
		int idx_guide =  im_w + im_h * image_width + b * image_width * image_height;
		
		float f_gr_d = (input_guide[idx_guide] + 0.5f) / sr;
		float f_gr_w = (im_w + 0.5f) / ss;
		float f_gr_h = (im_h + 0.5f) / ss;

		int front = static_cast<int>(floor(input_guide[idx_guide] / sr - 0.5f));
		int back = front + 1;

		int left = static_cast<int>(floor(im_w * 1.0 / ss - 0.5f));
		int right = left + 1;

		int up = static_cast<int>(floor(im_h * 1.0 / ss - 0.5f));
		int down = up + 1;

		int sw = channel_size;
		int su = channel_size * grid_depth;
		int sv = channel_size * grid_depth * grid_width;
		int sb = channel_size * grid_depth * grid_width * grid_height;

		float sum_color = 0.0f;
		for (int vv = up; vv <= down; vv++)
		{
			if (vv < 0 || vv > grid_height - 1)
				continue;
			float wy = weight_z((vv + 0.5f) - f_gr_h);
			for (int uu = left; uu <= right; uu++)
			{
				if (uu < 0 || uu > grid_width - 1)
					continue;
				float wx = weight_z((uu + 0.5f) - f_gr_w);
				for (int ww = front; ww <= back; ww++)
				{
					if (ww < 0 || ww > grid_depth - 1)
						continue;
					float wz = weight_z((ww + 0.5f) - f_gr_d);
					float wt = wy * wx * wz;
					
					float part2 = 0;
					int left_xx = static_cast<int>(ceil((uu - 0.5f) * ss));
					int right_xx = left_xx + 2 * ss + 2;
					int up_yy = static_cast<int>(ceil((vv - 0.5f) * ss));
					int down_yy = up_yy + 2 * ss + 2;

					for (int yy = up_yy; yy < down_yy; yy++)
					{
						if (yy < 0 || yy > image_height - 1)
							continue;
						for (int xx = left_xx; xx < right_xx; xx++)
						{
							if (xx < 0 || xx > image_width - 1)
								continue;
							int idx_guide_in = xx + yy * image_width + b * image_width * image_height;
							float f_xx = (xx + 0.5f) / ss;
							float f_yy = (yy + 0.5f) / ss;
							float f_zz = (input_guide[idx_guide_in] + 0.5f) / sr;

							float wyy = weight_z((vv + 0.5f) - f_yy);
							float wxx = weight_z((uu + 0.5f) - f_xx);
							float wzz = weight_z((ww + 0.5f) - f_zz);
							part2 += wyy * wxx * wzz;
						}
					}
					int idx_grid = c + ww * sw + uu * su + vv * sv + b * sb;
					// printf("i=%d, vv=%d, uu=%d, ww=%d, lt=%d, rt=%d, up=%d, dwn=%d\n", i, vv, uu, ww, left_xx, right_xx, up_yy,down_yy);
					if (abs(part2) > 0)
						sum_color += backprop[idx_grid] * wt / part2;
				}
			}
		}
		if (abs(sum_color) > 100)
		{
			printf("Create::Big Image Grad = %.4f\n", sum_color);
			image_grad[i] = 0;
		}
		else
		{
			image_grad[i] = sum_color;
		}
	}
}



__global__ void CreateBiGridGuideGradKernel(
	int num_guide_pixels,
	const float* input_image, 
	const float* input_guide,
	const float* backprop,
	const int batch_size,
	const int image_height,
	const int image_width,
	const int channel_size,
	const int grid_height, 
	const int grid_width, 
	const int grid_depth,
	const int sr,
	float* guide_grad
) {
	CUDA_KERNEL_LOOP(i, num_guide_pixels)
	{
		int ss = image_width / grid_width;

		int gd_w = i % image_width;
		int gd_h = (i / image_width) % image_height;
		int b = (i / (image_width * image_height)) % batch_size;
		
		float f_gr_d = (input_guide[i] + 0.5f) / sr;
		float f_gr_w = (gd_w + 0.5f) / ss;
		float f_gr_h = (gd_h + 0.5f) / ss;

		int front = static_cast<int>(floor(input_guide[i] / sr - 0.5f));
		int back = front + 1;

		int left = static_cast<int>(floor(gd_w * 1.0 / ss - 0.5f));
		int right = left + 1;

		int up = static_cast<int>(floor(gd_h * 1.0 / ss - 0.5f));
		int down = up + 1;

		// printf("gd_h=%d,gd_w=%d,front=%d,left=%d, up=%d\n", gd_h, gd_w, front, left, up);
		int sw = channel_size;
		int su = channel_size * grid_depth;
		int sv = channel_size * grid_depth * grid_width;
		int sb_uvw = channel_size * grid_depth * grid_width * grid_height;

		int sx = channel_size;
		int sy = channel_size * image_width;
		int sb_xy = channel_size * image_width * image_height;

		float sum_color = 0.0f;
		for (int cc = 0; cc < channel_size; cc++)
		{
			float chan_color = 0.0f;
			int idx_image = cc + gd_w * sx + gd_h * sy + b * sb_xy;
			for (int vv = up; vv <= down; vv++)
			{
				if (vv < 0 || vv > grid_height - 1)
					continue;
				for (int uu = left; uu <= right; uu++)
				{
					if (uu < 0 || uu > grid_width - 1)
						continue;
					for (int ww = front; ww <= back; ww++)
					{
						if (ww < 0 || ww > grid_depth - 1)
							continue;
						int left_xx = static_cast<int>(ceil((uu - 0.5f) * ss));
						int right_xx = left_xx + 2 * ss + 1;
						int up_yy = static_cast<int>(ceil((vv - 0.5f) * ss));
						int down_yy = up_yy + 2 * ss + 1;

						// printf("%d,%d,%d =%d\n", vv, uu, ww, left_xx);
						
						float wx = weight_z((uu + 0.5f) - f_gr_w);
						float wy = weight_z((vv + 0.5f) - f_gr_h);
						float dwz = d_weight_z((ww + 0.5f) - f_gr_d) / sr;
						
						float part1 = wy * wx * dwz * input_image[idx_image];
						float part3 = wy * wx * dwz;
						float part2 = 0.0f;
						float part4 = 0.0f;

						for (int xx = left_xx; xx < right_xx; xx++)
						{
							if (xx < 0 || xx > image_width - 1)
								continue;
							for (int yy = up_yy; yy < down_yy; yy++)
							{
								if (yy < 0 || yy > image_height - 1)
									continue;
								int i_guide = xx + image_width * yy + b * image_width * image_height;
								int i_image = cc + xx * sx + yy * sy + b * sb_xy;

								float f_xx = (xx + 0.5f) / ss;
								float f_yy = (yy + 0.5f) / ss;
								float f_zz = (input_guide[i_guide] + 0.5f) / sr;

								float wyy = weight_z((vv + 0.5f) - f_yy);
								float wxx = weight_z((uu + 0.5f) - f_xx);
								float wzz = weight_z((ww + 0.5f) - f_zz);

								float wtt = wyy * wxx * wzz;
								part2 += wtt;
								part4 += wtt * input_image[i_image];
							}
						}
						int idx_grid = cc + ww * sw + uu * su + vv * sv + b * sb_uvw;

						float derv = 0;
						if (abs(part2) > 0)
							derv = (part1 * part2 - part3 * part4) / (part2 * part2);
						chan_color += backprop[idx_grid] * derv;
						// printf("%d,%d,%d,%d, %.4f\n", i, vv, uu, ww, part1);
					}
				}
			}
			sum_color += chan_color;
		}
		if (abs(sum_color) > 100)
		{
			printf("Create::Big Guide Grad = %.4f\n", sum_color);
			guide_grad[i] = 0;
		}
		else
		{
			guide_grad[i] = sum_color;
		}
	}
}

#ifdef TRAINING
__global__ void SliceBiGridKernel(
	int subpixel_count,
	const float* input_grid,
	const float* input_guide,
	const int batch_size,
	const int image_height,
	const int image_width,
	const int channel_size,
	const int grid_height, 
	const int grid_width, 
	const int grid_depth,
	const int sr,
	float* output_image
)
{
	CUDA_KERNEL_LOOP(i, subpixel_count)
	{	
		int ss = image_width / grid_width;

		int c = i % channel_size;
		int im_w = (i / channel_size) % image_width;
		int im_h = (i / (channel_size * image_width)) % image_height;
		int b = (i / (channel_size * image_width * image_height)) % batch_size;
		
		int idx_guide =  im_w + im_h * image_width + b * image_width * image_height;
		
		float f_gr_d = (input_guide[idx_guide] + 0.5f) / sr;
		float f_gr_w = (im_w + 0.5f) / ss;
		float f_gr_h = (im_h + 0.5f) / ss;

		int front = static_cast<int>(floor(input_guide[idx_guide] / sr - 0.5f));
		int back = front + 1;

		int left = static_cast<int>(floor(im_w * 1.0f / ss - 0.5f));
		int right = left + 1;

		int up = static_cast<int>(floor(im_h * 1.0f / ss - 0.5f));
		int down = up + 1;

		int sw = channel_size;
		int su = channel_size * grid_depth;
		int sv = channel_size * grid_depth * grid_width;
		int sb = channel_size * grid_depth * grid_width * grid_height;

		float sum_color = 0.0f;
		float sum_count = 0.0f;
		for (int vv = up; vv <= down; vv++)
		{
			if (vv < 0 || vv > grid_height - 1)
				continue;
			float wy = weight_z((vv + 0.5f) - f_gr_h);
			for (int uu = left; uu <= right; uu++)
			{
				if (uu < 0 || uu > grid_width - 1)
					continue;
				float wx = weight_z((uu + 0.5f) - f_gr_w);
				for (int ww = front - 2; ww < back + 2; ww++)
				{
					if (ww < 0 || ww > grid_depth - 1)
						continue;
					float wz = weight_z((ww + 0.5f) - f_gr_d);
					int idx_grid = c + ww * sw + uu * su + vv * sv + b * sb;
					if (input_grid[idx_grid] > 0.0f)
					{
						float wt =  wy * wx * wz;
						sum_color += input_grid[idx_grid] * wt;
						sum_count += wt;
					}
				}
			}
		}
		output_image[i] = abs(sum_count) > 0.0f ? sum_color / sum_count: 0.0f;
	}
}
#else
__global__ void SliceBiGridKernel(
	int subpixel_count,
	const __half* input_grid,
	const __half* input_guide,
	const int batch_size,
	const int image_height,
	const int image_width,
	const int channel_size,
	const int grid_height, 
	const int grid_width, 
	const int grid_depth,
	const int sr,
	__half* output_image
)

{
	CUDA_KERNEL_LOOP(i, subpixel_count)
	{	
		int ss = image_width / grid_width;

		int c = i % channel_size;
		int im_w = (i / channel_size) % image_width;
		int im_h = (i / (channel_size * image_width)) % image_height;
		int b = (i / (channel_size * image_width * image_height)) % batch_size;
		
		int idx_guide =  im_w + im_h * image_width + b * image_width * image_height;
		
		__half f_gr_d = __hdiv(__hadd(input_guide[idx_guide], __float2half(0.5f)), __float2half(static_cast<float>(sr)));
		__half f_gr_w = __hdiv(__hadd(__float2half(static_cast<float>(im_w)), __float2half(0.5f)), __float2half(static_cast<float>(ss)));
		__half f_gr_h = __hdiv(__hadd(__float2half(static_cast<float>(im_h)), __float2half(0.5f)), __float2half(static_cast<float>(ss)));

		// int front = static_cast<int>(floor(input_guide[idx_guide] / sr - 0.5f));
		int front = static_cast<int>(__half2float(hfloor(__hsub(__hdiv(input_guide[idx_guide], __float2half(static_cast<float>(sr))), __float2half(0.5f)))));
		int back = front + 1;
		
		// int left = static_cast<int>(floor(im_w * 1.0f / ss - 0.5f));
		int left = static_cast<int>(__half2float(hfloor(__hsub(__hdiv(__hmul(__float2half(static_cast<float>(im_w)), __float2half(1.0f)), __float2half(static_cast<float>(ss))), __float2half(0.5f)))));
		int right = left + 1;
		
		// int up = static_cast<int>(floor(im_h * 1.0f / ss - 0.5f));
		int up = static_cast<int>(__half2float(hfloor(__hsub(__hdiv(__hmul(__float2half(static_cast<float>(im_h)), __float2half(1.0f)), __float2half(static_cast<float>(ss))), __float2half(0.5f)))));
		int down = up + 1;

		int sw = channel_size;
		int su = channel_size * grid_depth;
		int sv = channel_size * grid_depth * grid_width;
		int sb = channel_size * grid_depth * grid_width * grid_height;

		__half sum_color = __float2half(0.0f);
		__half sum_count = __float2half(0.0f);
		for (int vv = up; vv <= down; vv++)
		{
			if (vv < 0 || vv > grid_height - 1)
				continue;
			// at::Half wy = weight_z((vv + 0.5) - f_gr_h);
			__half wy = weight_z(__hsub(__hadd(__float2half(static_cast<float>(vv)), __float2half(0.5f)), f_gr_h));
			for (int uu = left; uu <= right; uu++)
			{
				if (uu < 0 || uu > grid_width - 1)
					continue;
				// __half wx = weight_z((uu + 0.5) - f_gr_w);
				__half wx = weight_z(__hsub(__hadd(__float2half(static_cast<float>(uu)), __float2half(0.5f)), f_gr_w));
				for (int ww = front - 2; ww < back + 2; ww++)
				{
					if (ww < 0 || ww > grid_depth - 1)
						continue;
					// at::Half wz = weight_z((ww + 0.5) - f_gr_d);
					__half wz = weight_z(__hsub(__hadd(__float2half(static_cast<float>(ww)), __float2half(0.5f)), f_gr_d));
					int idx_grid = c + ww * sw + uu * su + vv * sv + b * sb;
					if (__hgt(input_grid[idx_grid], __float2half(0.0f)))
					{
						// __half wt =  wy * wx * wz;
						__half wt = __hmul(__hmul(wy, wx), wz);
						// sum_color += input_grid[idx_grid] * wt;
						sum_color = __hadd(sum_color, __hmul(input_grid[idx_grid], wt));
						// sum_count += wt;
						sum_count = __hadd(sum_count, wt);
					}
				}
			}
		}
		output_image[i] = __hgt(__habs(sum_count), __float2half(0.0f)) ? __hdiv(sum_color, sum_count) : __float2half(0.0f);
	}
}
#endif

__global__ void SliceBiGridGridGradKernel(
	int num_grid_points,
	const float* input_grid,
	const float* input_guide,
	const float* backprop,
	const int batch_size,
	const int image_height,
	const int image_width,
	const int channel_size,
	const int grid_height, 
	const int grid_width, 
	const int grid_depth,
	const int sr,
	float* const grid_grad
) {
	CUDA_KERNEL_LOOP(i, num_grid_points)
	{
		float sum_color = 0;
		if (input_grid[i] > 0.0f)
		{
			int ss = image_width / grid_width;
			int c = i % channel_size;
			int d = (i / channel_size) % grid_depth;
			int w = (i / (channel_size * grid_depth)) % grid_width;
			int h = (i / (channel_size * grid_depth * grid_width)) % grid_height;
			int b = (i / (channel_size * grid_depth * grid_width * grid_height)) % batch_size;

			float f_w = w + 0.5f;
			float f_h = h + 0.5f;
			float f_d = d + 0.5f;

			int left = static_cast<int>(ceil((w - 0.5f) * ss));
			int right = left + 2 * ss;
			int up = static_cast<int>(ceil((h - 0.5f) * ss));
			int down = up + 2 * ss;

			// printf("h=%d, w=%d, d=%d, c=%d, left=%d, right=%d, up=%d, down=%d\n",
			// 			h,w,d,c,left,right, up,down);
			int sw = channel_size;
			int su = channel_size * grid_depth;
			int sv = channel_size * grid_depth * grid_width;
			int sb_uvw = channel_size * grid_depth * grid_width * grid_height;

			int sx = channel_size;
			int sy = channel_size * image_width;
			int sb_xy = channel_size * image_width * image_height;

			for (int yy = up; yy < down; yy++)
			{
				if (yy < 0 || yy > image_height - 1)
					continue;
				for (int xx = left; xx < right; xx++)
				{
					if (xx < 0 || xx > image_width - 1)
						continue;
					int idx_guide = xx + yy * image_width + b * image_width * image_height;
					int idx_image = c + xx * sx + yy * sy + b * sb_xy;

					float wy = weight_z((yy + 0.5f) / ss - f_h);
					float wx = weight_z((xx + 0.5f) / ss - f_w);
					float wz = weight_z((input_guide[idx_guide] + 0.5f) / sr - f_d);
					float wt =  wy * wx * wz;

					// printf("h=%d, w=%d, d=%d, c=%d, yy=%d, xx=%d, wy=%.4f, wx=%.4f, wz=%.4f\n",
					// 	h,w,d,c,yy,xx, wy,wx,wz);
					int front_ww = static_cast<int>(floor(input_guide[idx_guide] / sr - 0.5f));
					int back_ww = front_ww + 1;

					int left_uu = static_cast<int>(floor(xx * 1.0f / ss - 0.5f));
					int right_uu = left_uu + 1;

					int up_vv = static_cast<int>(floor(yy * 1.0f / ss - 0.5f));
					int down_vv = up_vv + 1;

					float part2 = 0.0f;
					for (int ww = front_ww; ww <= back_ww; ww++)
					{
						if (ww < 0 || ww > grid_depth - 1)
							continue;
						for(int vv = up_vv; vv <= down_vv; vv++)
						{
							if (vv < 0 || vv > grid_height - 1)
								continue;
							for (int uu = left_uu; uu <= right_uu; uu++)
							{
								if (uu < 0 || uu > grid_width - 1)
									continue;
								float f_uu = uu + 0.5f;
								float f_vv = vv + 0.5f;
								float f_ww = ww + 0.5f;

								float wyy = weight_z((yy + 0.5f) / ss - f_vv);
								float wxx = weight_z((xx + 0.5f) / ss - f_uu);
								float wzz = weight_z((input_guide[idx_guide] + 0.5f) / sr - f_ww);
								
								int idx_grid = c + ww * sw + uu * su + vv * sv + b * sb_uvw;
								if (input_grid[idx_grid] > 0.0f)
									part2 += wyy * wxx * wzz;
							}
						}
					}
					// printf("h=%d, w=%d, d=%d, c=%d, yy=%d, xx=%d, p1=%.4f, p2=%.4f\n",
					// 	h,w,d,c,yy,xx, wt, part2);
					if (abs(part2) > 0)
					{
						sum_color += backprop[idx_image] * wt / part2;
					}
				}
			}
			if (abs(sum_color) > 100)
			{
				printf("Slice::Big Grid Grad = %.4f\n", sum_color);
				grid_grad[i] = 0;
			}
			else
			{
				grid_grad[i] = sum_color;
			}
		} else {
			grid_grad[i] = 0;
		}
	}
}

__global__ void SliceBiGridGuideGradKernel(
	int num_guide_pixels,
	const float* input_grid, 
	const float* input_guide,
	const float* backprop,
	const int batch_size,
	const int image_height,
	const int image_width,
	const int channel_size,
	const int grid_height, 
	const int grid_width, 
	const int grid_depth,
	const int sr,
	float* guide_grad
) {
	CUDA_KERNEL_LOOP(i, num_guide_pixels)
	{
		int ss = image_width / grid_width;

		int gd_w = i % image_width;
		int gd_h = (i / image_width) % image_height;
		int b = (i / (image_width * image_height)) % batch_size;
				
		float f_gr_d = (input_guide[i] + 0.5f) / sr;
		float f_gr_w = (gd_w + 0.5f) / ss;
		float f_gr_h = (gd_h + 0.5f) / ss;

		int front = static_cast<int>(floor(input_guide[i] / sr - 0.5f));
		int back = front + 1;

		int left = static_cast<int>(floor(gd_w * 1.0f / ss - 0.5f));
		int right = left + 1;

		int up = static_cast<int>(floor(gd_h * 1.0f / ss - 0.5f));
		int down = up + 1;

		int sw = channel_size;
		int su = channel_size * grid_depth;
		int sv = channel_size * grid_depth * grid_width;
		int sb_uvw = channel_size * grid_depth * grid_width * grid_height;

		int sx = channel_size;
		int sy = channel_size * image_width;
		int sb_xy = channel_size * image_width * image_height;
		
		float sum_color = 0.0f;
		for (int cc = 0; cc < channel_size; cc++)
		{
			float part1 = 0.0f;
			float part2 = 0.0f;
			float part3 = 0.0f;
			float part4 = 0.0f;
			for (int vv = up; vv <= down; vv++)
			{
				if (vv < 0 || vv > grid_height - 1)
					continue;
				for (int uu = left; uu <= right; uu++)
				{
					if (uu < 0 || uu > grid_width - 1)
						continue;
					for (int ww = 0; ww < grid_depth; ww++)
					{
						if (ww < 0 || ww > grid_depth - 1)
							continue;
						float wy = weight_z((vv + 0.5f) - f_gr_h);
						float wx = weight_z((uu + 0.5f) - f_gr_w);
						float wz = weight_z((ww + 0.5f) - f_gr_d);
						float dwz = d_weight_z((ww + 0.5f) - f_gr_d) / sr;

						int idx_grid = cc + ww * sw + uu * su + vv * sv + b * sb_uvw;

						if (input_grid[idx_grid] > 0.0f)
						{
							float dwt = wy * wx * dwz;
							float wt = wy * wx * wz;
							part1 += dwt * input_grid[idx_grid];
							part2 += wt;
							part3 += dwt;
							part4 += wt * input_grid[idx_grid];
						}
					}
				}
			}
			int idx_image = cc + gd_w * sx + gd_h * sy + b * sb_xy;
			float derv = 0;
			if (abs(part2) > 0)
			{
				derv = (part1 * part2 - part3 * part4) / (part2 * part2);
			}
			sum_color += derv * backprop[idx_image];
		}
		if (abs(sum_color) > 100)
		{
			printf("Slice::Big Guide Grad = %.4f\n", sum_color);
			guide_grad[i] = 0;
		}
		else
		{
			guide_grad[i] = sum_color;
		}
	}
}

#ifdef TRAINING
void create_grid_kernel_launcher
(
	const float *input_image,
	const float *input_guide,
	const int batch_size,
	const int image_height,
	const int image_width,
	const int channel_size,
	const int grid_height,
	const int grid_width,
	const int grid_depth,
	const int sr,
	float *out_color, // changed to non const
	torch::Device &device,
	const torch::Dtype &type
)
{
	int64_t num_grid_points = batch_size * grid_height * grid_width * channel_size;

	torch::Device current_device(torch::kCUDA, 0);

	if (current_device == device && num_grid_points > 0)
	{		// int64_t num_blocks = at::cuda::detail::GET_BLOCKS(num_grid_points);
		int64_t num_blocks = at::cuda::detail::GET_BLOCKS(num_grid_points, static_cast<uint64_t>(512));

#if defined(USE_ROCM) || defined(TORCH_USE_HIP_DSA)
    const hipStream_t stream = at::cuda::getCurrentHIPStream();
#else
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#endif

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        type, "CreateBiGridKernel", ([&] {
        // CreateBiGridKernel<<<num_blocks, at::cuda::detail::CUDA_NUM_THREADS, 0, stream>>>
		CreateBiGridKernel<<<num_blocks, static_cast<uint64_t>(512), 0, stream>>>
		(
        	num_grid_points,
			input_image,
			input_guide,
			batch_size,
        	image_height,
			image_width,
			channel_size,
			grid_height,
        	grid_width,
			grid_depth,
			sr,
			out_color
		);
        }));
	}
}
#else
void create_grid_kernel_launcher
(
	const at::Half *input_image,
	const at::Half *input_guide,
	const int batch_size,
	const int image_height,
	const int image_width,
	const int channel_size,
	const int grid_height,
	const int grid_width,
	const int grid_depth,
	const int sr,
	at::Half *out_color, // changed to non const
	torch::Device &device,
	const torch::Dtype &type
)
{
	int64_t num_grid_points = batch_size * grid_height * grid_width * channel_size;

	torch::Device current_device(torch::kCUDA, 0);

	if (current_device == device && num_grid_points > 0)
	{		// int64_t num_blocks = at::cuda::detail::GET_BLOCKS(num_grid_points);
		int64_t num_blocks = at::cuda::detail::GET_BLOCKS(num_grid_points, static_cast<uint64_t>(512));

#if defined(USE_ROCM) || defined(TORCH_USE_HIP_DSA)
    const hipStream_t stream = at::cuda::getCurrentHIPStream();
#else
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#endif

	const __half *__hinput_image = reinterpret_cast<const __half *>(input_image);
	const __half *__hinput_guide = reinterpret_cast<const __half *>(input_guide);
	__half *__hout_color = reinterpret_cast<__half *>(out_color);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        type, "CreateBiGridKernel", ([&] {
        // CreateBiGridKernel<<<num_blocks, at::cuda::detail::CUDA_NUM_THREADS, 0, stream>>>
		CreateBiGridKernel<<<num_blocks, static_cast<uint64_t>(512), 0, stream>>>
		(
        	num_grid_points,
			__hinput_image,
			__hinput_guide,
			batch_size,
        	image_height,
			image_width,
			channel_size,
			grid_height,
        	grid_width,
			grid_depth,
			sr,
			__hout_color
		);
        }));
	}
}
#endif

void create_grid_grad_kernel_launcher
(
    const float* input_image,
	const float* input_guide,
    const float* backprop,
	const int batch_size,
	const int image_height,
	const int image_width,
	const int channel_size,
	const int grid_height, 
	const int grid_width, 
	const int grid_depth,
	const int sr,
	float* const image_grad,
	float* const guide_grad,
    torch::Device& device,
    const torch::Dtype& type
)
{

	int64_t num_image_pixels = batch_size*image_height*image_width*channel_size;
	int64_t num_guide_pixels = batch_size*image_height*image_width;

	torch::Device current_device(torch::kCUDA, 0);

	if (current_device == device && num_image_pixels > 0 && num_guide_pixels > 0)
	{
    	// int64_t num_blocks = at::cuda::detail::GET_BLOCKS(num_image_pixels);
		int64_t num_blocks = at::cuda::detail::GET_BLOCKS(num_image_pixels, static_cast<uint64_t>(512));

#if defined(USE_ROCM) || defined(TORCH_USE_HIP_DSA)
    	const hipStream_t stream = at::cuda::getCurrentHIPStream();
#else
    	const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#endif

		AT_DISPATCH_FLOATING_TYPES_AND_HALF(
		    type, "CreateBiGridImageGradKernel", ([&] {
		    // CreateBiGridImageGradKernel<<<num_blocks, at::cuda::detail::CUDA_NUM_THREADS, 0, stream>>>
			CreateBiGridImageGradKernel<<<num_blocks, static_cast<uint64_t>(512), 0, stream>>>
			(
		    	num_image_pixels,
				input_image,
				input_guide,
				backprop,
				batch_size,
		    	image_height,
				image_width,
				channel_size,
				grid_height,
		    	grid_width,
				grid_depth,
				sr,
				image_grad
			);
		}));
  

    	AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    	    type, "CreateBiGridGuideGradKernel", ([&] {
    	    // CreateBiGridGuideGradKernel<<<num_blocks, at::cuda::detail::CUDA_NUM_THREADS, 0, stream>>>
			CreateBiGridGuideGradKernel<<<num_blocks, static_cast<uint64_t>(512), 0, stream>>>
			(
    	    	num_guide_pixels,
				input_image,
				input_guide,
				backprop,
				batch_size,
    	    	image_height,
				image_width,
				channel_size,
				grid_height,
    	    	grid_width,
				grid_depth,
				sr,
				guide_grad
			);
    	}));
  }
}

#ifdef TRAINING
void slice_grid_kernel_launcher
(
    const float* input_grid,
    const float* input_guide,
    const int batch_size,
    const int image_height,
    const int image_width,
    const int channel_size,
    const int grid_height,
    const int grid_width,
    const int grid_depth,
    const int sr,
    float* const output_image,
    torch::Device& device,
    const torch::Dtype& type
)
{
	int64_t subpixel_count = batch_size * image_height * image_width * channel_size;

	if(subpixel_count > 0)
	{
		torch::Device current_device(torch::kCUDA, 0);

		int64_t num_blocks = at::cuda::detail::GET_BLOCKS(subpixel_count, static_cast<uint64_t>(512));

#if defined(USE_ROCM) || defined(TORCH_USE_HIP_DSA)
    	const hipStream_t stream = at::cuda::getCurrentHIPStream();
#else
    	const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#endif

		AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        	type, "SliceBiGridKernel", ([&] {
        	// SliceBiGridKernel<<<num_blocks, at::cuda::detail::CUDA_NUM_THREADS, 0, stream>>>
			SliceBiGridKernel<<<num_blocks, static_cast<uint64_t>(512), 0, stream>>>
			(
				subpixel_count,
				input_grid,
				input_guide,
				batch_size,
				image_height,
				image_width,
				channel_size,
				grid_height,
				grid_width,
				grid_depth,
				sr,
				output_image
			);
    	}));

	}
}
#else
void slice_grid_kernel_launcher
(
    const at::Half* input_grid,
    const at::Half* input_guide,
    const int batch_size,
    const int image_height,
    const int image_width,
    const int channel_size,
    const int grid_height,
    const int grid_width,
    const int grid_depth,
    const int sr,
    at::Half* const output_image,
    torch::Device& device,
    const torch::Dtype& type
)
{
	int64_t subpixel_count = batch_size * image_height * image_width * channel_size;

	if(subpixel_count > 0)
	{
		torch::Device current_device(torch::kCUDA, 0);

		int64_t num_blocks = at::cuda::detail::GET_BLOCKS(subpixel_count, static_cast<uint64_t>(512));

#if defined(USE_ROCM) || defined(TORCH_USE_HIP_DSA)
    	const hipStream_t stream = at::cuda::getCurrentHIPStream();
#else
    	const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#endif

		const __half *__hinput_grid = reinterpret_cast<const __half *>(input_grid);
		const __half *__hinput_guide = reinterpret_cast<const __half *>(input_guide);
		__half * const __houtput_image = reinterpret_cast<__half * const>(output_image);

		AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        	type, "SliceBiGridKernel", ([&] {
        	// SliceBiGridKernel<<<num_blocks, at::cuda::detail::CUDA_NUM_THREADS, 0, stream>>>
			SliceBiGridKernel<<<num_blocks, static_cast<uint64_t>(512), 0, stream>>>
			(
				subpixel_count,
				__hinput_grid,
				__hinput_guide,
				batch_size,
				image_height,
				image_width,
				channel_size,
				grid_height,
				grid_width,
				grid_depth,
				sr,
				__houtput_image
			);
    	}));

	}
}
#endif

void slice_grid_grad_kernel_launcher
(
	const float* input_grid,
    const float* input_guide,
	const float* backprop,
    const int batch_size,
    const int image_height,
    const int image_width,
    const int channel_size,
    const int grid_height,
    const int grid_width,
    const int grid_depth,
    const int sr,
    float* const grid_grad,
	float* const guide_grad,
    torch::Device& device,
    const torch::Dtype& type
)
{
	int64_t num_grid_points = batch_size * grid_height * grid_width * grid_depth * channel_size;
	int64_t num_guide_pixels = batch_size * image_height * image_width;

	torch::Device current_device(torch::kCUDA, 0);

	if (current_device == device && num_grid_points > 0 && num_guide_pixels > 0)
	{
    	// int64_t num_blocks = at::cuda::detail::GET_BLOCKS(num_grid_points);

		int64_t num_blocks = at::cuda::detail::GET_BLOCKS(num_grid_points, static_cast<uint64_t>(512)); // done for the V100 GPU on the CHPC cluster

#if defined(USE_ROCM) || defined(TORCH_USE_HIP_DSA)
    	const hipStream_t stream = at::cuda::getCurrentHIPStream();
#else
    	const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#endif

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        type, "SliceBiGridGridGradKernel", ([&] {
        //SliceBiGridGridGradKernel<<<num_blocks, at::cuda::detail::CUDA_NUM_THREADS, 0, stream>>>
		SliceBiGridGridGradKernel<<<num_blocks, static_cast<uint64_t>(512), 0, stream>>>
		(
			num_grid_points,
			input_grid,
			input_guide,
			backprop,
			batch_size,
			image_height,
			image_width,
			channel_size,
			grid_height,
			grid_width,
			grid_depth,
			sr,
			grid_grad
		);
    }));
  

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        type, "SliceBiGridGuideGradKernel", ([&] {
        // SliceBiGridGuideGradKernel<<<num_blocks, at::cuda::detail::CUDA_NUM_THREADS, 0, stream>>>
		SliceBiGridGuideGradKernel<<<num_blocks, static_cast<uint64_t>(512), 0, stream>>>
		(
			num_guide_pixels,
			input_grid,
			input_guide,
			backprop,
			batch_size,
			image_height,
			image_width,
			channel_size,
			grid_height,
			grid_width,
			grid_depth,
			sr,
			guide_grad
		);
    }));

  }
}
