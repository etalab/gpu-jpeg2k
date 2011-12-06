/**
 * @file quantization.cu
 *
 * @author Milosz Ciznicki
 * @date 06-09-2010
 */

#include "../types/image_types.h"
#include "../misc/cuda_errors.h"
extern "C" {
	#include "quantizer.h"
	#include "quantization.h"
	#include "../misc/memory_management.cuh"
	#include "../print_info/print_info.h"
}

/**
 * @brief Subband quantization.
 *
 * @param idata Input tile_comp_data.
 * @param size Width and height of subbnad.
 * @param step_size Step size(deltab).
 */
__global__
void subband_quantization_lossy(type_data *idata, int2 isize, int *odata, int2 osize, int2 cblk_size, const float convert_factor)
{
	int cblk_sum_size = cblk_size.x * cblk_size.y;
	int block_id = blockIdx.x + blockIdx.y * gridDim.x;
	int i = threadIdx.x;
	int j = threadIdx.y;
	int n = i + blockIdx.x * cblk_size.x;
	int m = j + blockIdx.y * cblk_size.y;
	int in = n + m * isize.x;
	int out = i + j * cblk_size.x + (block_id) * cblk_sum_size;
	int cache_value;

	while(j < cblk_size.y && m < osize.y)
	{
		while(i < cblk_size.x &&  n < osize.x)
		{
			cache_value = idata[in] * convert_factor;
			odata[out] = cache_value < 0 ? (1 << 31) | ((int)(-cache_value)) : cache_value;
//			odata[out] = (idata[in] * convert_factor) < 0 ? (1 << 31) | ((int)(-(idata[in] * convert_factor))) : (idata[in] * convert_factor);
			i += BLOCKSIZEX;
			n = i + blockIdx.x * cblk_size.x;
			in = n + m * isize.x;
			out = i + j * cblk_size.x + (block_id) * cblk_sum_size;
		}
		i = threadIdx.x;
		j += BLOCKSIZEY;
		n = i + blockIdx.x * cblk_size.x;
		m = j + blockIdx.y * cblk_size.y;
		in = n + m * isize.x;
		out = i + j * cblk_size.x + (block_id) * cblk_sum_size;
	}
}

/**
 * @brief Subband quantization.
 *
 * @param idata Input tile_comp_data.
 * @param size Width and height of subbnad.
 * @param step_size Step size(deltab).
 */
__global__
void subband_quantization_lossless(type_data *idata, int2 isize, int *odata, int2 osize, int2 cblk_size, const int shift_bits)
{
	int cblk_sum_size = cblk_size.x * cblk_size.y;
	int block_id = blockIdx.x + blockIdx.y * gridDim.x;
	int i = threadIdx.x;
	int j = threadIdx.y;
	int n = i + blockIdx.x * cblk_size.x;
	int m = j + blockIdx.y * cblk_size.y;
	int in = n + m * isize.x;
	int out = i + j * cblk_size.x + (block_id) * cblk_sum_size;
	int cache_value;

	while(j < cblk_size.y && m < osize.y)
	{
		while(i < cblk_size.x &&  n < osize.x)
		{
			cache_value = ((int)idata[in]) << shift_bits;
			odata[out] = cache_value < 0 ? (1 << 31) | (-cache_value) : cache_value;
//			odata[out] = (idata[in] * convert_factor) < 0 ? (1 << 31) | ((int)(-(idata[in] * convert_factor))) : (idata[in] * convert_factor);
			i += BLOCKSIZEX;
			n = i + blockIdx.x * cblk_size.x;
			in = n + m * isize.x;
			out = i + j * cblk_size.x + (block_id) * cblk_sum_size;
		}
		i = threadIdx.x;
		j += BLOCKSIZEY;
		n = i + blockIdx.x * cblk_size.x;
		m = j + blockIdx.y * cblk_size.y;
		in = n + m * isize.x;
		out = i + j * cblk_size.x + (block_id) * cblk_sum_size;
	}
}

/**
 * @brief Subband quantization.
 *
 * @param subband Input subband.
 * @param size Width and height of subbnad.
 */
type_subband *quantization(type_subband *sb)
{
//	println_start(INFO);

	dim3 blocks;
	dim3 threads;
	int i;
	type_codeblock *cblk;

	type_res_lvl *res_lvl = sb->parent_res_lvl;
	type_tile_comp *tile_comp = res_lvl->parent_tile_comp;
	type_image *img = tile_comp->parent_tile->parent_img;

	/* Base step size */
	float base_step = /*BASE_STEP*/(float)1 / (float)(1 << (img->num_range_bits - 1));
	int guard_bits = GUARD_BITS;
	/* Relative quantization step size. Step size is signaled relative to the wavelet coefficient bit depth. */
	float relative_step_size;
	int shift_bits;

	/* Lossy */
	if(img->wavelet_type)
	{
		/* The number of magnitude bits in the integer representation of the quantized data */
		sb->mag_bits = guard_bits - 1 + res_lvl->dec_lvl_no - (int)(log(base_step) / log(2));
		/* Relative quantization step size. Step size is signaled relative to the wavelet coefficient bit depth. */
		relative_step_size = base_step / (1 << res_lvl->dec_lvl_no);
		shift_bits = 31 - sb->mag_bits;

		/* Calculate step size and convert factor for quantization */
		relative_step_size = convert_from_exp_mantissa(convert_to_exp_mantissa(relative_step_size));
		sb->step_size = ((1L << (img->num_range_bits + get_exp_subband_gain(sb->orient))) * relative_step_size);
		sb->convert_factor = (1.0f / sb->step_size) * (1 << (shift_bits));

//		println_var(INFO, "Lossy mag_bits:%d convert_factor:%f shift_bits:%d relative_step_size:%f step_size:%f subband_gain:%d", sb->mag_bits, sb->convert_factor, shift_bits, relative_step_size, sb->step_size, get_exp_subband_gain(sb->orient));

	} else /* Lossless */
	{
		/* K_b_max = B - 1 + X_b + G */
		sb->mag_bits = img->num_range_bits - 1 + get_exp_subband_gain(sb->orient) + guard_bits;
		shift_bits = 31 - sb->mag_bits;
		sb->convert_factor = 1 << shift_bits;

//		println_var(INFO, "Lossless mag_bits:%d convert_factor:%f shift_bits:%d subband_gain:%d", sb->mag_bits, sb->convert_factor, shift_bits, get_exp_subband_gain(sb->orient));
	}
//	println_var(INFO, "%d %d %d %d %d %d", sb->width, sb->num_cblks, sb->num_xcblks, sb->num_ycblks, tile_comp->cblk_w, tile_comp->cblk_h);
	/* Allocate aligned memory for linear codeblocks data */
	cuda_d_allocate_mem((void **) &(sb->cblks_data_d), sb->num_cblks * tile_comp->cblk_w * tile_comp->cblk_h * sizeof(int));

	/* Input and output data */
	type_data *idata = tile_comp->img_data_d + sb->tlx + sb->tly * tile_comp->width;
	int *odata = sb->cblks_data_d;
	int2 isize = make_int2(tile_comp->width, tile_comp->height);
	int2 osize = make_int2(sb->width, sb->height);
	int2 cblk_size = make_int2(tile_comp->cblk_w, tile_comp->cblk_h);

	/* Number of blocks for parallel reduction */
	blocks = dim3(sb->num_xcblks , sb->num_ycblks);
	/* Number of threads for parallel reduction */
	threads = dim3(BLOCKSIZEX, BLOCKSIZEY);

//	println_var(INFO, "blocks.x:%d blocks.y:%d threads.x:%d threads.y:%d", blocks.x, blocks.y, threads.x, threads.y);
//	println_var(INFO, "isize.x:%d isize.y:%d osize.x:%d osize.y:%d cblk_size.x:%d cblk_size.y:%d", isize.x, isize.y, osize.x, osize.y, cblk_size.x, cblk_size.y);
	if(img->wavelet_type)
	{
		subband_quantization_lossy<<<blocks, threads>>>(idata, isize, odata, osize, cblk_size, sb->convert_factor);
	} else
	{
		subband_quantization_lossless<<<blocks, threads>>>(idata, isize, odata, osize, cblk_size, shift_bits);
	}

	cudaThreadSynchronize();

	checkCUDAError("quantization");

	/* Update code blocks pointer to data */
	for(i = 0; i < sb->num_cblks; i++)
	{
		cblk = &(sb->cblks[i]);
		cblk->data_d = sb->cblks_data_d + cblk->cblk_no * tile_comp->cblk_w * tile_comp->cblk_h;
	}

//	println_end(INFO);

	return sb;
}

/**
 * @brief Do quantization for every subbands from tile.
 * @param tile
 */
void quantize_tile(type_tile *tile)
{
//	println_start(INFO);

//	start_measure();

	type_image *img = tile->parent_img;
	type_tile_comp *tile_comp;
	type_res_lvl *res_lvl;
	type_subband *sb;
	int i, j, k;

	for(i = 0; i < img->num_components; i++)
	{
		tile_comp = &(tile->tile_comp[i]);
		for(j = 0; j < tile_comp->num_rlvls; j++)
		{
			res_lvl = &(tile_comp->res_lvls[j]);
			for(k = 0; k < res_lvl->num_subbands; k++)
			{
				sb = &(res_lvl->subbands[k]);
				quantization(sb);
			}
		}
		cuda_d_free(tile_comp->img_data_d);
	}

//	stop_measure(INFO);

//	println_end(INFO);
}

/**
 * @brief Do quantization for every subbands from tile.
 * @param tile
 */
/*void quantize_tile_dbg(type_tile *tile)
{
	println_start(INFO);

	start_measure();

	type_image *img = tile->parent_img;
	type_tile_comp *tile_comp;
	type_res_lvl *res_lvl;
	type_subband *sb;
	int i, j, k;

	int x = 0, y = 0;
	int size = tile->width * tile->height * sizeof(type_data);
	type_data *buff = (type_data*)malloc(size);

	for(i = 0; i < img->num_components; i++)
	{
		tile_comp = &(tile->tile_comp[i]);
		cuda_memcpy_dth(tile_comp->img_data_d, buff, size);

		for(y = 0; y < tile_comp->height; y++)
		{
			for(x = 0; x < tile_comp->width; x++)
			{
				printf("%f, ", buff[x + y * tile_comp->width]);
			}
		}
		printf("\n");

		for(j = 0; j < tile_comp->num_rlvls; j++)
		{
			res_lvl = &(tile_comp->res_lvls[j]);
			for(k = 0; k < res_lvl->num_subbands; k++)
			{
				sb = &(res_lvl->subbands[k]);
				quantization(sb);
			}
		}
		cuda_d_free(tile_comp->img_data_d);
	}

	stop_measure(INFO);

	println_end(INFO);
}*/

/**
 * @brief Subband quantization.
 *
 * @param subband Input subband.
 * @param size Width and height of subbnad.
 */
//type_subband *quantization_dbg(type_subband *sb)
//{
////	println_start(INFO);
//
//	dim3 blocks;
//	dim3 threads;
//	int i;
//	type_codeblock *cblk;
//
//	type_res_lvl *res_lvl = sb->parent_res_lvl;
//	type_tile_comp *tile_comp = res_lvl->parent_tile_comp;
//	type_image *img = tile_comp->parent_tile->parent_img;
//
//	/* Base step size */
//	float base_step = BASE_STEP;
//	int guard_bits = GUARD_BITS;
//	/* Relative quantization step size. Step size is signaled relative to the wavelet coefficient bit depth. */
//	float relative_step_size;
//	int shift_bits;
//
//	/* Lossy */
//	if(img->wavelet_type)
//	{
//		/* The number of magnitude bits in the integer representation of the quantized data */
//		sb->mag_bits = guard_bits - 1 + res_lvl->dec_lvl_no - (int)(log(base_step) / log(2));
//		/* Relative quantization step size. Step size is signaled relative to the wavelet coefficient bit depth. */
//		relative_step_size = base_step / (1 << res_lvl->dec_lvl_no);
//		shift_bits = 31 - sb->mag_bits;
//
//		/* Calculate step size and convert factor for quantization */
//		relative_step_size = convert_from_exp_mantissa(convert_to_exp_mantissa(relative_step_size));
//		sb->step_size = ((1L << (img->nom_range_bits + get_exp_subband_gain(sb->orient))) * relative_step_size);
//		sb->convert_factor = (1.0f / sb->step_size) * (1 << (shift_bits));
//
////		println_var(INFO, "Lossy mag_bits:%d convert_factor:%f shift_bits:%d relative_step_size:%f step_size:%f subband_gain:%d", sb->mag_bits, sb->convert_factor, shift_bits, relative_step_size, sb->step_size, get_exp_subband_gain(sb->orient));
//
//	} else /* Lossless */
//	{
//		/* K_b_max = B - 1 + X_b + G */
//		sb->mag_bits = img->nom_range_bits - 1 + get_exp_subband_gain(sb->orient) + guard_bits;
//		shift_bits = 31 - sb->mag_bits;
//		sb->convert_factor = 1 << shift_bits;
//
////		println_var(INFO, "Lossless mag_bits:%d convert_factor:%f shift_bits:%d subband_gain:%d", sb->mag_bits, sb->convert_factor, shift_bits, get_exp_subband_gain(sb->orient));
//	}
////	println_var(INFO, "%d %d %d %d %d %d", sb->width, sb->num_cblks, tile_comp->cblk_w, tile_comp->cblk_h, sizeof(long int), sizeof(size_t));
//	/* Allocate aligned memory for linear codeblocks data */
//	cuda_d_allocate_mem((void **) &(sb->cblks_data_d), sb->num_cblks * tile_comp->cblk_w * tile_comp->cblk_h * sizeof(int));
//
//	/* Input and output data */
//	type_data *idata = tile_comp->img_data_d + sb->tlx + sb->tly * tile_comp->height;
//	int *odata = sb->cblks_data_d;
//	int2 isize = make_int2(tile_comp->width, tile_comp->height);
//	int2 osize = make_int2(sb->width, sb->height);
//	int2 cblk_size = make_int2(tile_comp->cblk_w, tile_comp->cblk_h);
//
//	/* Number of blocks for parallel reduction */
//	blocks = dim3(sb->num_xcblks , sb->num_ycblks);
//	/* Number of threads for parallel reduction */
//	threads = dim3(BLOCKSIZEX, BLOCKSIZEY);
//
////	quantization_dbg(idata, osize.x * osize.y * sizeof(type_data), osize.x, osize.y);
//
////	println_var(INFO, "blocks.x:%d blocks.y:%d threads.x:%d threads.y:%d", blocks.x, blocks.y, threads.x, threads.y);
////	println_var(INFO, "isize.x:%d isize.y:%d osize.x:%d osize.y:%d cblk_size.x:%d cblk_size.y:%d", isize.x, isize.y, osize.x, osize.y, cblk_size.x, cblk_size.y);
//	if(img->wavelet_type)
//	{
//		subband_quantization_lossy<<<blocks, threads>>>(idata, isize, odata, osize, cblk_size, sb->convert_factor);
//	} else
//	{
//		subband_quantization_lossless<<<blocks, threads>>>(idata, isize, odata, osize, cblk_size, shift_bits);
//	}
//
//	cudaThreadSynchronize();
//
//	checkCUDAError("quantization");
//
//	int size = tile_comp->cblk_w * tile_comp->cblk_h * sizeof(int);
//	int *buff = (int*)malloc(size);
//	int x = 0, y = 0;
//
//	/* Update code blocks pointer to data */
//	for(i = 0; i < sb->num_cblks; i++)
//	{
//		cblk = &(sb->cblks[i]);
//		cblk->data_d = sb->cblks_data_d + cblk->cblk_no * tile_comp->cblk_w * tile_comp->cblk_h;
//
//		cuda_memcpy_dth(cblk->data_d, buff, size);
//
//		printf("ulx:%d uly:%d magbits:%d orient:%d con:%f\n", sb->tlx, sb->tly, sb->mag_bits, sb->orient, sb->convert_factor);
//
//		for(y = cblk->height - 1; y >= 0; y--)
//		{
//			for(x = cblk->width - 1; x >= 0; x--)
//			{
//				printf("%d, ", buff[x + y * tile_comp->cblk_w]);
//			}
//		}
//		printf("\n");
//	}
//
////	println_end(INFO);
//
//	return sb;
//}

/*void quantization_dbg(type_data *idata, int size, int w, int h)
{
	type_data *buff = (type_data*)malloc(size);
	int x = 0, y = 0;

	cuda_memcpy_dth(idata, buff, size);

	for(y = h - 1; y >= 0; y--)
	{
		for(x = w - 1; x >= 0; x--)
		{
			printf("%f, ", buff[x + y * w]);
		}
	}
	printf("\n");
	free(buff);
}*/
