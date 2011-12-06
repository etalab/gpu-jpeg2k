/*
 * shift.c
 *
 *  Created on: Nov 30, 2011
 *      Author: miloszc
 */

#include <stdint.h>
#include "shift.h"

#define BLOCK_SIZE 16
#define TILE_SIZEX 32
#define TILE_SIZEY 32

void __global__ mean_shift_kernel(type_data *idata, const uint16_t width, const uint16_t height, const type_data mean) {
	int i = threadIdx.x;
	int j = threadIdx.y;
	int n = i + blockIdx.x * TILE_SIZEX;
	int m = j + blockIdx.y * TILE_SIZEY;
	int idx = n + m * width;

	while(j < TILE_SIZEY && m < height)
	{
		while(i < TILE_SIZEX && n < width)
		{
			idata[idx] = idata[idx] - mean;
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
}

void shit(type_data *idata, const uint16_t w, const uint16_t h, const type_data mean) {
	dim3 dimGrid((w + (TILE_SIZEX - 1))/TILE_SIZEX, (h + (TILE_SIZEY - 1))/TILE_SIZEY);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	mean_shift_kernel<<<dimGrid, dimBlock>>>(idata, w, h, mean);
}
