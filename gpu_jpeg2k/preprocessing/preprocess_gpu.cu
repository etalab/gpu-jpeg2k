/**
 * @file preprocess_gpu.cu
 *
 * @brief Performes image preprocessing.
 *
 * It is the first step of the encoder workflow and the last one of the decoder workflow.
 * Changes image's color space to YUV.
 * Includes modes for lossy and lossless colorspace transformation.
 *
 * @author Jakub Misiorny <misiorny@man.poznan.pl>
 * @author Milosz Ciznicki
 */

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "../types/image_types.h"
#include "../misc/cuda_errors.h"
#include "../misc/memory_management.cuh"

extern "C" {
	#include "preprocess_gpu.h"
	#include "../print_info/print_info.h"
}

#include "preprocessing_constants.cuh"

/**
 * @brief CUDA kernel for DC level shifting coder.
 *
 * It performs dc level shifting to centralize data around 0. Doesn't use
 * any sophisticated algorithms for this, just subtracts 128. (so the data range is [-128 ; 128] ).
 *
 * @param img The image data.
 * @param size Number of pixels in each component (width x height).
 * @param level_shift Level shift.
 */
void __global__ fdc_level_shift_kernel(type_data *idata, const uint16_t width, const uint16_t height, const int level_shift) {
	int i = threadIdx.x;
	int j = threadIdx.y;
	int n = i + blockIdx.x * TILE_SIZEX;
	int m = j + blockIdx.y * TILE_SIZEY;
	int idx = n + m * width;

	while(j < TILE_SIZEY && m < height)
	{
		while(i < TILE_SIZEX && n < width)
		{
			idata[idx] = idata[idx] - (1 << level_shift);
			i += BLOCK_SIZE;
			n = i + blockIdx.x * TILE_SIZEX;
			idx = n + m * width;
		}
		i = threadIdx.x;
		j += BLOCK_SIZE;
		n = i + blockIdx.x * TILE_SIZEX;
		m = j + blockIdx.y * TILE_SIZEY;
		idx = n + m * width;
	}

	/*int idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

	if(idx < size) {
		img[idx] -= 1 << level_shift;
	}*/
}

int __device__ clamp_val(int val, int min, int max)
{
	if(val < min)
		return min;
	if(val > max)
		return max;
	return val;
}

/**
 * @brief CUDA kernel for DC level shifting decoder.
 *
 * It performs dc level shifting to centralize data around 0. Doesn't use
 * any sophisticated algorithms for this, just adds 128. (so the data range is [-128 ; 128] ).
 *
 * @param img The image data.
 * @param size Number of pixels in each component (width x height).
 * @param level_shift Level shift.
 */
void __global__ idc_level_shift_kernel(type_data *idata, const uint16_t width, const uint16_t height, const int level_shift, const int min, const int max) {
	int i = threadIdx.x;
	int j = threadIdx.y;
	int n = i + blockIdx.x * TILE_SIZEX;
	int m = j + blockIdx.y * TILE_SIZEY;
	int idx = n + m * width;
	int cache;

	while(j < TILE_SIZEY && m < height)
	{
		while(i < TILE_SIZEX && n < width)
		{
			cache = idata[idx] + (1 << level_shift);
			idata[idx] = clamp_val(cache, min, max);
//			idata[idx] = idata[idx] + (1 << level_shift);
			i += BLOCK_SIZE;
			n = i + blockIdx.x * TILE_SIZEX;
			idx = n + m * width;
		}
		i = threadIdx.x;
		j += BLOCK_SIZE;
		n = i + blockIdx.x * TILE_SIZEX;
		m = j + blockIdx.y * TILE_SIZEY;
		idx = n + m * width;
	}

	/*int idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

	if(idx < size) {
		img[idx] -= 1 << level_shift;
	}*/
}

/**
 * @brief CUDA kernel for the Reversible Color Transformation (lossless) coder.
 *
 * Before colorspace transformation it performs dc level shifting to centralize data around 0. Doesn't use
 * any sophisticated algorithms for this, just subtracts 128. (so the data range is [-128 ; 128] ).
 *
 * @param img_r 1D array with RED component of the image.
 * @param img_g 1D array with GREEN component of the image.
 * @param img_b 1D array with BLUE component of the image.
 * @param size Number of pixels in each component (width x height).
 */
void __global__ rct_kernel(type_data *img_r, type_data *img_g, type_data *img_b, const uint16_t width, const uint16_t height, const int level_shift) {
	int i = threadIdx.x;
	int j = threadIdx.y;
	int n = i + blockIdx.x * TILE_SIZEX;
	int m = j + blockIdx.y * TILE_SIZEY;
	int idx = n + m * width;
	int r, g, b;
	int y, u, v;

	while(j < TILE_SIZEY && m < height)
	{
		while(i < TILE_SIZEX && n < width)
		{
			b = img_b[idx] - (1 << level_shift);
			g = img_g[idx] - (1 << level_shift);
			r = img_r[idx] - (1 << level_shift);

			y = (r + 2*g + b)>>2;
			u = b - g;
			v = r - g;

			img_r[idx] = (type_data)y;
			img_b[idx] = (type_data)u;
			img_g[idx] = (type_data)v;

			i += BLOCK_SIZE;
			n = i + blockIdx.x * TILE_SIZEX;
			idx = n + m * width;
		}
		i = threadIdx.x;
		j += BLOCK_SIZE;
		n = i + blockIdx.x * TILE_SIZEX;
		m = j + blockIdx.y * TILE_SIZEY;
		idx = n + m * width;
	}

	/*int idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

	if(idx < size) {
		int r, g, b;
		int y, u, v;

		b = img_b[idx] - 128;
		g = img_g[idx] - 128;
		r = img_r[idx] - 128;

		y = (r + 2*g + b)>>2;
		u = b - g;
		v = r - g;

		img_b[idx] = (type_data)y;
		img_g[idx] = (type_data)u;
		img_r[idx] = (type_data)v;
	}*/
}

/**
 * @brief CUDA kernel for the Reversible Color Transformation (lossless) decoder.
 *
 *
 * After colorspace transformation it performs dc level shifting to shift data back to it's unsigned form,
 * just adds 128. (so the data range is [0 ; 256] ).
 *
 * @param img_r 1D array with V component of the image.
 * @param img_g 1D array with U component of the image.
 * @param img_b 1D array with Y component of the image.
 * @param size Number of pixels in each component (width x height).
 */
//void __global__ tcr_kernel(type_data *img_r, type_data *img_g, type_data *img_b, long int size) {
void __global__ tcr_kernel(type_data *img_r, type_data *img_g, type_data *img_b, const uint16_t width, const uint16_t height, const int level_shift, const int min, const int max) {
		int i = threadIdx.x;
		int j = threadIdx.y;
		int n = i + blockIdx.x * TILE_SIZEX;
		int m = j + blockIdx.y * TILE_SIZEY;
		int idx = n + m * width;
		int r, g, b;
		int y, u, v;

		while(j < TILE_SIZEY && m < height)
		{
			while(i < TILE_SIZEX && n < width)
			{
				y = img_r[idx];
				u = img_g[idx];
				v = img_b[idx];


				g = y - ((v + u)>>2);
				r = (v + g);
				b = (u + g);

				b = (type_data)b + (1 << level_shift);
				g = (type_data)g + (1 << level_shift);
				r = (type_data)r + (1 << level_shift);

				img_r[idx] = clamp_val(r, min, max);
				img_b[idx] = clamp_val(g, min, max);
				img_g[idx] = clamp_val(b, min, max);

//				img_r[idx] = (type_data)b + (1 << level_shift);
//				img_b[idx] = (type_data)g + (1 << level_shift);
//				img_g[idx] = (type_data)r + (1 << level_shift);

				i += BLOCK_SIZE;
				n = i + blockIdx.x * TILE_SIZEX;
				idx = n + m * width;
			}

			i = threadIdx.x;
			j += BLOCK_SIZE;
			n = i + blockIdx.x * TILE_SIZEX;
			m = j + blockIdx.y * TILE_SIZEY;
			idx = n + m * width;
		}

/*
	int idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

	if(idx < size) {
		type_data r, g, b;
		type_data y, u, v;

		y = img_b[idx];
		u = img_g[idx];
		v = img_r[idx];

		g = y - floor((u + v) / 4);
		r = (v + g);
		b = (u + g);

		img_b[idx] = b + 128;
		img_g[idx] = g + 128;
		img_r[idx] = r + 128;
	}
*/
}

/**
 * @brief CUDA kernel for the Irreversible Color Transformation (lossy) coder.
 *
 * Before colorspace transformation it performs dc level shifting to centralize data around 0. Doesn't use
 * any sophisticated algorithms for this, just subtracts 128. (so the data range is [-128 ; 128] ).
 *
 * @param img_r 1D array with RED component of the image.
 * @param img_g 1D array with GREEN component of the image.
 * @param img_b 1D array with BLUE component of the image.
 * @param size Number of pixels in each component (width x height).
 */
void __global__ ict_kernel(type_data *img_r, type_data *img_g, type_data *img_b, const uint16_t width, const uint16_t height, const int level_shift) {
	int i = threadIdx.x;
	int j = threadIdx.y;
	int n = i + blockIdx.x * TILE_SIZEX;
	int m = j + blockIdx.y * TILE_SIZEY;
	int idx = n + m * width;
	type_data r, g, b;
	type_data y, u, v;

	while(j < TILE_SIZEY && m < height)
	{
		while(i < TILE_SIZEX && n < width)
		{
			b = img_r[idx] - (1 << level_shift);
			g = img_g[idx] - (1 << level_shift);
			r = img_b[idx] - (1 << level_shift);

			y = Wr*r + Wg*g + Wb*b;
			u = -0.16875f * r - 0.33126f * g + 0.5f * b;
//			u = (Umax * ((b - y) / (1 - Wb)));
			v = 0.5f * r - 0.41869f * g - 0.08131f * b;
//			v = (Vmax * ((r - y) / (1 - Wr)));

			img_r[idx] = y;
			img_g[idx] = u;
			img_b[idx] = v;

/*			img_r[idx] = y;
			img_g[idx] = u;
			img_b[idx] = v;*/

			i += BLOCK_SIZE;
			n = i + blockIdx.x * TILE_SIZEX;
			idx = n + m * width;
		}
		i = threadIdx.x;
		j += BLOCK_SIZE;
		n = i + blockIdx.x * TILE_SIZEX;
		m = j + blockIdx.y * TILE_SIZEY;
		idx = n + m * width;
	}
	/*int idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

    if(idx < size) {
		type_data r, g, b;
		type_data y, u, v;

		b = img_b[idx] - 128;
		g = img_g[idx] - 128;
		r = img_r[idx] - 128;

		y = Wr*r + Wg*g + Wb*b;
		u = (Umax * ((b - y) / (1 - Wb)));
		v = (Vmax * ((r - y) / (1 - Wr)));

		img_b[idx] = y;
		img_g[idx] = u;
		img_r[idx] = v;
	}*/
}

/**
 * @brief CUDA kernel for the Irreversible Color Transformation (lossy) decoder.
 *
 *
 * After colorspace transformation it performs dc level shifting to shift data back to it's unsigned form,
 * just adds 128. (so the data range is [0 ; 256] ).
 *
 * @param img_r 1D array with V component of the image.
 * @param img_g 1D array with U component of the image.
 * @param img_b 1D array with Y component of the image.
 * @param size Number of pixels in each component (width x height).
 */
void __global__ tci_kernel(type_data *img_r, type_data *img_g, type_data *img_b, const uint16_t width, const uint16_t height, const int level_shift, const int min, const int max) {
	int i = threadIdx.x;
	int j = threadIdx.y;
	int n = i + blockIdx.x * TILE_SIZEX;
	int m = j + blockIdx.y * TILE_SIZEY;
	int idx = n + m * width;
	type_data r, g, b;
	type_data y, u, v;

	while(j < TILE_SIZEY && m < height)
	{
		while(i < TILE_SIZEX && n < width)
		{
			y = img_b[idx];
			u = img_g[idx];
			v = img_r[idx];

			type_data r_tmp = v*( (1 - Wr)/Vmax );
			type_data b_tmp = u*( (1 - Wb)/Umax );

			r = y + r_tmp;
			b = y + b_tmp;
			g = y - (Wb/Wg) * r_tmp - (Wr/Wg) * b_tmp;

			b = (type_data)b + (1 << level_shift);
			g = (type_data)g + (1 << level_shift);
			r = (type_data)r + (1 << level_shift);

			img_b[idx] = clamp_val(b, min, max);
			img_g[idx] = clamp_val(g, min, max);
			img_r[idx] = clamp_val(r, min, max);

//			img_b[idx] = b + (1 << level_shift);
//			img_g[idx] = g + (1 << level_shift);
//			img_r[idx] = r + (1 << level_shift);

			i += BLOCK_SIZE;
			n = i + blockIdx.x * TILE_SIZEX;
			idx = n + m * width;
		}
		i = threadIdx.x;
		j += BLOCK_SIZE;
		n = i + blockIdx.x * TILE_SIZEX;
		m = j + blockIdx.y * TILE_SIZEY;
		idx = n + m * width;
	}

/*
	int idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

	if(idx < size) {
		type_data r, g, b;
		type_data y, u, v;

		y = img_b[idx];
		u = img_g[idx];
		v = img_r[idx];

		type_data r_tmp = v*( (1 - Wr)/Vmax );
		type_data b_tmp = u*( (1 - Wb)/Umax );

		r = y + r_tmp;
		b = y + b_tmp;
		g = y - (Wb/Wg) * r_tmp - (Wr/Wg) * b_tmp;
	
		img_b[idx] = b + 128;
		img_g[idx] = g + 128;
		img_r[idx] = r + 128;
	}
*/
}

/**
 * Lossy color transformation RGB -> YCbCr (the data can be in any colorspace, though. But the coefficients are designed for RGB). Coder.
 *
 * @param img Image to be color-transformated
 * @return Returns the color transformated image. It's just the pointer to the same structure passed in img parameter.
 */
int color_coder_lossy(type_image *img) {
	return color_trans_gpu(img, ICT);
}

/**
 * Lossy color transformation YCbCr -> RGB. Decoder of the color_coder_lossy output.
 *
 * @param img Image to be color-transformated
 * @return Returns the color transformated image. It's just the pointer to the same structure passed in img parameter.
 */
int color_decoder_lossy(type_image *img) {
	return color_trans_gpu(img, TCI);
}

/**
 * Lossless color transformation RGB -> YUV (the data can be in any colorspace, though. But the coefficients are designed for RGB). Coder.
 *
 * @param img Image to be color-transformated
 * @return Returns the color transformated image. It's just the pointer to the same structure passed in img parameter.
 */
int color_coder_lossless(type_image *img) {
	return color_trans_gpu(img, RCT);
}


/**
 * Lossless color transformation YUV -> RGB. Decoder of the color_coder_lossless output.
 *
 * @param img Image to be color-transformated
 * @return Returns the color transformated image. It's just the pointer to the same structure passed in img parameter.
 */
int color_decoder_lossless(type_image *img) {
	return color_trans_gpu(img, TCR);
}

/**
 * @brief Main function of color transformation flow. Should not be called directly though. Use four wrapper functions color_[de]coder_loss[y|less] instead.
 *
 * @param img type_image to will be transformed.
 * @param type Type of color transformation that should be performed. The types are detailed in color_trans_type.
 *
 *
 * @return Returns 0 on sucess.
 */
int color_trans_gpu(type_image *img, color_trans_type type) {
	if(img->num_components != 3) {
		println(INFO, "Error: Color transformation not possible. The number of components != 3.");
		exit(0);
	}

	//CUDA timing apparatus
	#ifdef COMPUTE_TIME
		cudaEvent_t kernel_start, kernel_stop;

		cudaEventCreate(&kernel_start);
		cudaEventCreate(&kernel_stop);
	#endif

	int tile_size = 0, i;
	type_tile *tile;
	type_data *comp_a, *comp_b, *comp_c;

	int level_shift = img->num_range_bits - 1;

	int min = img->sign == SIGNED ? -(1 << (img->num_range_bits - 1)) : 0;
	int max = img->sign == SIGNED ? (1 << (img->num_range_bits - 1)) - 1 : (1 << img->num_range_bits) - 1;

	switch(type) {
	case RCT:
//		println_var(INFO, "start: RCT");
		#ifdef COMPUTE_TIME
			cudaEventRecord(kernel_start, 0);
		#endif

		for(i = 0; i < img->num_tiles; i++) {
			tile = &(img->tile[i]);
			comp_a = (&(tile->tile_comp[0]))->img_data_d;
			comp_b = (&(tile->tile_comp[1]))->img_data_d;
			comp_c = (&(tile->tile_comp[2]))->img_data_d;

			dim3 dimGrid((tile->width + (TILE_SIZEX - 1))/TILE_SIZEX, (tile->height + (TILE_SIZEY - 1))/TILE_SIZEY);
			dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

//			printf("%d\n", level_shift);
/*			int blocks = ( tile_size / (BLOCK_SIZE * BLOCK_SIZE)) + 1;
			dim3 dimGrid(blocks);
			dim3 dimBlock(BLOCK_SIZE* BLOCK_SIZE);*/

			rct_kernel<<<dimGrid, dimBlock>>>( comp_a, comp_b, comp_c, tile->width, tile->height, level_shift );
		}

		#ifdef COMPUTE_TIME
			cudaEventRecord(kernel_stop, 0);
			cudaEventSynchronize( kernel_stop );
		#endif
		break;

	case TCR:
//		println_var(INFO, "start: TCR");
		#ifdef COMPUTE_TIME
			cudaEventRecord(kernel_start, 0);
		#endif

			for(i = 0; i < img->num_tiles; i++) {
				tile = &(img->tile[i]);
				comp_a = (&(tile->tile_comp[0]))->img_data_d;
				comp_b = (&(tile->tile_comp[1]))->img_data_d;
				comp_c = (&(tile->tile_comp[2]))->img_data_d;

				int blocks = ( tile_size / (BLOCK_SIZE * BLOCK_SIZE)) + 1;
				dim3 dimGrid(blocks);
				dim3 dimBlock(BLOCK_SIZE* BLOCK_SIZE);

				tcr_kernel<<< dimGrid, dimBlock, 0>>>( comp_a, comp_b, comp_c, tile->width, tile->height, level_shift, min, max);
			}
		
		#ifdef COMPUTE_TIME
			cudaEventRecord(kernel_stop, 0);
			cudaEventSynchronize( kernel_stop );
		#endif
		break;

	case ICT:
//		println_var(INFO, "start: ICT");
		#ifdef COMPUTE_TIME
			cudaEventRecord(kernel_start, 0);
		#endif

		for(i = 0; i < img->num_tiles; i++) {
			tile = &(img->tile[i]);
			comp_a = (&(tile->tile_comp[0]))->img_data_d;
			comp_b = (&(tile->tile_comp[1]))->img_data_d;
			comp_c = (&(tile->tile_comp[2]))->img_data_d;

			dim3 dimGrid((tile->width + (TILE_SIZEX - 1))/TILE_SIZEX, (tile->height + (TILE_SIZEY - 1))/TILE_SIZEY);
			dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
			int level_shift = img->num_range_bits - 1;
			/*int blocks = ( tile_size / (BLOCK_SIZE * BLOCK_SIZE)) + 1;
			dim3 dimGrid(blocks);
			dim3 dimBlock(BLOCK_SIZE* BLOCK_SIZE);*/

			ict_kernel<<< dimGrid, dimBlock>>>( comp_a, comp_b, comp_c, tile->width, tile->height, level_shift );
		}

		#ifdef COMPUTE_TIME
			cudaEventRecord(kernel_stop, 0);
			cudaEventSynchronize( kernel_stop );
		#endif
		break;

	case TCI:
//		println_var(INFO, "start: TCI");
		#ifdef COMPUTE_TIME
		cudaEventRecord(kernel_start, 0);
		#endif


		for(i = 0; i < img->num_tiles; i++) {
			tile = &(img->tile[i]);
			tile_size = tile->width * tile->height;
			comp_a = (&(tile->tile_comp[0]))->img_data_d;
			comp_b = (&(tile->tile_comp[1]))->img_data_d;
			comp_c = (&(tile->tile_comp[2]))->img_data_d;

			int blocks = ( tile_size / (BLOCK_SIZE * BLOCK_SIZE)) + 1;
			dim3 dimGrid(blocks);
			dim3 dimBlock(BLOCK_SIZE* BLOCK_SIZE);

			tci_kernel<<< dimGrid, dimBlock, 0>>>( comp_a, comp_b, comp_c, tile->width, tile->height, level_shift, min, max);
		}

		#ifdef COMPUTE_TIME
			cudaEventRecord(kernel_stop, 0);
			cudaEventSynchronize( kernel_stop );
		#endif
		break;

	}

	checkCUDAError("color_trans_gpu");

	float kernel_time;
	#ifdef COMPUTE_TIME
		cudaEventElapsedTime( &kernel_time, kernel_start, kernel_stop );
		cudaEventDestroy( kernel_start );
		cudaEventDestroy( kernel_stop );
		printf("\t\tkernel: %.2f [ms]\n", kernel_time);
	#endif
//	println_end(INFO);
	return 0;
}

void dc_level_shifting(type_image *img, int sign)
{
	int i, j;
	type_tile *tile;
	type_data *idata;
	int min = 0;
	int max = (1 << img->num_range_bits) - 1;

//	start_measure();

	for(i = 0; i < img->num_tiles; i++)
	{
		tile = &(img->tile[i]);
		for(j = 0; j < img->num_components; j++)
		{
			idata = (&(tile->tile_comp[j]))->img_data_d;

			dim3 dimGrid((tile->width + (TILE_SIZEX - 1))/TILE_SIZEX, (tile->height + (TILE_SIZEY - 1))/TILE_SIZEY);
			dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
			int level_shift = img->num_range_bits - 1;

			if(sign < 0)
			{
				fdc_level_shift_kernel<<<dimGrid, dimBlock>>>( idata, tile->width, tile->height, level_shift);
			} else
			{
				idc_level_shift_kernel<<<dimGrid, dimBlock>>>( idata, tile->width, tile->height, level_shift, min, max);
			}

			cudaThreadSynchronize();
			checkCUDAError("dc_level_shifting");
		}
	}

//	stop_measure(INFO);
}

/**
 * @brief Forward DC level shifting.
 * @param img
 * @param type
 */
void fdc_level_shifting(type_image *img)
{
	dc_level_shifting(img, -1);
}

/**
 * @brief Inverse DC level shifting.
 * @param img
 * @param type
 */
void idc_level_shifting(type_image *img)
{
	dc_level_shifting(img, 1);
}
