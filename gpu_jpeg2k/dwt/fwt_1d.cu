extern "C"
{
#include "fwt_1d.h"
#include "../print_info/print_info.h"
#include "../misc/memory_management.cuh"
#include "../misc/cuda_errors.h"
}

#define BLOCK 256
#define BLOCKX 16
#define BLOCKY 16
#define SM_SIZE 256

/**
 * @defgroup 97Coeff 97 Coefficients.
 *
 * 97 Coefficients.
 *
 * @{
 */
const float a1 = -1.586134342f;
const float a2 = -0.05298011854f;
const float a3 = 0.8829110762f;
const float a4 = 0.4435068522f;
/** @} */

/**
 * @defgroup ScaleCoeff Scale coefficients.
 *
 * Scale coefficients.
 *
 * @{
 */
const float k = 1.230174104914f; // 1.230174104914
/** @} */

__device__ static void read_data(float *sm, int tidx, type_data** data, int w, int h, int num_components, short offset)
{
	const short p_offset_l = ((tidx < offset) ? (offset - tidx) /* left symmetric extension*/: -offset + tidx /* take normally pixels */);
	int p_offset;
	int pix = blockIdx.x + w * blockIdx.y;

	while (tidx < num_components + 2 * offset)
	{
		short p_offset_r = (num_components - 2) - (tidx - (num_components + offset)) /* right symmetric extension*/;

		p_offset = ((tidx >= num_components + offset) ? p_offset_r : p_offset_l);

		sm[tidx] = data[p_offset][pix];

		tidx += BLOCK;
	}
}

__device__ static void save_data(int tidx, type_data **data, int w, float *pix_neighborhood, int num_components)
{
	int pix = blockIdx.x + w * blockIdx.y;
	int high_pass = (num_components + 1) >> 1;

	while (2*tidx < num_components)
	{
		data[tidx][pix] = pix_neighborhood[4] / k;

		if (tidx + high_pass < num_components)
		{
			data[tidx + high_pass][pix] = k * pix_neighborhood[5];
		}
		tidx += BLOCK;
	}
}

/**
 * @brief Does lifting process.
 *
 * @param a Coefficient.
 * @param pix_neighborhood Array storing neighbor pixels.
 */
template<class T, unsigned int start, unsigned int end>
__device__
void process(const float a, T *pix_neighborhood)
{
#pragma unroll
	for (int i = start; i <= end; i += 2)
	{
		pix_neighborhood[i] += a * (pix_neighborhood[i - 1] + pix_neighborhood[i + 1]);
	}
}

__device__ void process_97(int tidx, float *pix_neighborhood, float *sm, int offset)
{
	// Read necessary data
#pragma unroll
	for (int i = 0; i < 9; i++)
	{
		/* Data start from offset */
		pix_neighborhood[i] = sm[tidx + i];
	}

	// Predict 1
	process<float, 1, 7> (a1, pix_neighborhood);

	// Update 1
	process<float, 2, 6> (a2, pix_neighborhood);

	// Predict 2
	process<float, 3, 5> (a3, pix_neighborhood);

	// Update 2
	process<float, 4, 4> (a4, pix_neighborhood);
}

__global__ void fwt_1d_kernel(type_data** data, int w, int h, int num_components, short offset)
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
		process_97(tidx2, pix_neighborhood, sm, offset);
		tidx2 += BLOCK;
	}
	__syncthreads();

	tidx = threadIdx.x;

	save_data(tidx, data, w, pix_neighborhood, num_components);

	/*	if (blockIdx.x >= w || blockIdx.y >= h || tidx >= num_components)
	 {
	 return;
	 }*/

}

void fwt_1d(type_image *img, int lvl)
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
		fprintf(stderr, "Cuda error: %s: %s.\n", "Error before fwt_1d_kernel", cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}

	type_tile *tile = &img->tile[0];
	type_tile_comp *tile_comp = &tile->tile_comp[0];

	/* Number of all thread blocks */
	dim3 grid_size = dim3(tile_comp-> width, tile_comp->height, 1);

//	printf("w:%d h:%d num_comp:%d\n", tile_comp->width, tile_comp->height, img->num_components);

	int n = img->num_components;
	int i;
	for(i = 0; i < lvl; ++i)
	{
		fwt_1d_kernel<<<grid_size, BLOCK>>>(data_pd, tile_comp->width, tile_comp->height, n, 4);

		n = (n + 1) >> 1;
	}

	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if( cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", "Error in fwt_1d_kernel", cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}

	free(data_p);
	cudaFree(data_pd);
}
