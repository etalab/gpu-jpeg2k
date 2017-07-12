/* 
Copyright 2009-2013 Poznan Supercomputing and Networking Center

Authors:
Milosz Ciznicki miloszc@man.poznan.pl

GPU JPEG2K is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GPU JPEG2K is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with GPU JPEG2K. If not, see <http://www.gnu.org/licenses/>.
*/
/**
 * @file iwt_1d.cu
 *
 * @author Milosz Ciznicki
 */

extern "C"
{
#include <stdio.h>
#include <string.h>
#include "iwt_1d.h"
#include "../print_info/print_info.h"
#include "../misc/memory_management.cuh"
#include "../misc/cuda_errors.h"
}

#define BLOCK 256
#define SM_SIZE 256

/**
 * @defgroup I97Coeff Inverse 97 Coefficients.
 *
 * Inverse 97 Coefficients.
 *
 * @{
 */
const float a1 = 1.586134342f;
const float a2 = 0.05298011854f;
const float a3 = -0.8829110762f;
const float a4 = -0.4435068522f;
/** @} */

/**
 * @defgroup IScaleCoeff Inverse scale coefficients.
 *
 * Inverse scale coefficients.
 *
 * @{
 */
const float k = 1.230174104914f;	// 1.230174104914
/** @} */

__device__ static void read_data(float *sm, int tidx, type_data** data, int w, int h, int num_components, short offset)
{
	const short p_offset_lowpass_l = ((tidx < offset) ? (offset - tidx) /* left symmetric extension*/: -offset + tidx /* take normally pixels */);
	const short p_offset_highpass_l = ((tidx < offset) ? (offset - 1 - tidx) /* left symmetric extension*/: -offset + tidx /* take normally pixels */);
	int p_offset_lowpass;
	int p_offset_highpass;
	int pix = blockIdx.x + w * blockIdx.y;
	int low_pass = (num_components + 1) >> 1;
	int high_pass = num_components - low_pass;
	int parity = (num_components & 1) == 0 ? 1 : 0;

	while (tidx < low_pass + 2 * offset)
	{
		short p_offset_lowpass_r = (low_pass - (2 - parity)) - (tidx - (low_pass + offset)) /* right symmetric extension*/;
		short p_offset_highpass_r = (high_pass - (1 + parity)) - (tidx - (high_pass + offset)) /* right symmetric extension*/;

		int tid_low = 2 * tidx;
		int tid_high = 2 * tidx + 1;

		p_offset_lowpass = ((tid_low >= num_components + 2 * offset) ? p_offset_lowpass_r : p_offset_lowpass_l);
		p_offset_highpass = ((tid_high >= num_components + 2 * offset) ? p_offset_highpass_r : p_offset_highpass_l);

		sm[tid_low] = k * data[p_offset_lowpass][pix];
		if(low_pass + p_offset_highpass < num_components)
			sm[tid_high] = data[low_pass + p_offset_highpass][pix] / k;

		tidx += BLOCK;
	}
}

__device__ static void save_data(int tidx, type_data **data, int w, float *pix_neighborhood, int num_components)
{
	int pix = blockIdx.x + w * blockIdx.y;

	while (tidx < num_components)
	{
		data[tidx][pix] = pix_neighborhood[4];
		if(tidx + 1 < num_components)
			data[tidx + 1][pix] = pix_neighborhood[5];

		tidx += BLOCK;
	}
}

/**
 * @brief Does inverse lifting process.
 *
 * @param a Coefficient.
 * @param pix_neighborhood Array storing neighbor pixels.
 */
template <class T, unsigned int start, unsigned int end>
__device__ void process(const float a, T *pix_neighborhood, int round)
{
#pragma unroll
	for(int i = start; i <= end; i+=2)
	{
		pix_neighborhood[i] += a * (pix_neighborhood[i-1] + pix_neighborhood[i+1] + round);
	}
}

__device__ void iprocess_97(int tidx, float *pix_neighborhood, float *sm, int offset)
{
	// Read necessary data
#pragma unroll
	for (int i = 0; i < 10; i++)
	{
		/* Data start from offset */
		pix_neighborhood[i] = sm[tidx + i];
	}

	// Update 2
	process<float, 2, 8> (a4, pix_neighborhood, 0);

	// Predict 2
	process<float, 3, 7> (a3, pix_neighborhood, 0);

	// Update 1
	process<float, 4, 6> (a2, pix_neighborhood, 0);

	// Predict 1
	process<float, 5, 5> (a1, pix_neighborhood, 0);
}

__global__ void iwt_1d_kernel(type_data** data, int w, int h, int num_components, short offset)
{
	__shared__ float sm[SM_SIZE]; // TODO provide space for offset (dynamic allocation)
	int tidx = threadIdx.x;

	read_data(sm, tidx, data, w, h, num_components, offset);
	__syncthreads();

	float pix_neighborhood[9];

	tidx = threadIdx.x;
	int tidx2 = threadIdx.x * 2;

	while (tidx2 < num_components)
	{
		iprocess_97(tidx2, pix_neighborhood, sm, offset);
		tidx2 += BLOCK;
	}
	__syncthreads();

	tidx = threadIdx.x;
	tidx2 = threadIdx.x * 2;

	save_data(tidx2, data, w, pix_neighborhood, num_components);
}

void iwt_1d(type_image *img, int lvl)
{
	type_data** data_pd;
	cudaMalloc(&data_pd, sizeof(type_data*) * img->num_components);

	type_data** data_p = (type_data**) calloc(img->num_components, sizeof(type_data*));
	for (int g = 0; g < img->num_components; ++g)
	{
		data_p[g] = img->tile[0].tile_comp[g].img_data_d;
	}
	cudaMemcpy(data_pd, data_p, sizeof(type_data*) * img->num_components, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", "Error before iwt_1d_kernel", cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}

	type_tile *tile = &img->tile[0];
	type_tile_comp *tile_comp = &tile->tile_comp[0];

	/* Number of all thread blocks */
	dim3 grid_size = dim3(tile_comp-> width, tile_comp->height, 1);

	printf("w:%d h:%d num_comp:%d\n", tile_comp->width, tile_comp->height, img->num_components);

	int n = img->num_components;
	int *parity = (int *)my_malloc(lvl * sizeof(int));
	memset(parity, 0, lvl * sizeof(int));
	int i;
	for(i = 0; i < lvl - 1; ++i)
	{
		parity[i] = ((n & 1) == 1) ? 1 : 0;
		n = (n + 1) >> 1;
	}

	printf("n:%d\n", n);

	for(i = 0; i < lvl; ++i)
	{
		iwt_1d_kernel<<<grid_size, BLOCK>>>(data_pd, tile_comp->width, tile_comp->height, n, 2);

		n = n * 2 - parity[lvl - 2 - i];
	}

	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if( cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", "Error in iwt_1d_kernel", cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}

	free(data_p);
	cudaFree(data_pd);
}
