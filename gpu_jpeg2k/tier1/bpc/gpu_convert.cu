/*
 * gpu_convert.cu
 *
 *  Created on: Feb 20, 2012
 *      Author: miloszc
 */

#include <stdio.h>
#include <assert.h>
#include "gpu_bpc.h"

#define POS_NEW_BITPOS 31
#define POS_NEW (1 << POS_NEW_BITPOS)
#define POS 0x1fff
#define EMPTY_BITPOS 30
#define EMPTY (1 << EMPTY_BITPOS)

//pos
//31		12-0
//pos_new	position

template <char Code_Block_Size_X>
__device__ void preprocessing(unsigned int cxds[][Code_Block_Size_X], unsigned int pos[][Code_Block_Size_X], int *blockVote, unsigned char &pass) {
//	int *blockVote_ = blockVote;
	if(TID == 0) atomicAnd(blockVote, 0);
	//blockVote = 0;
	__syncthreads();
	assert(*blockVote == 0);
	__syncthreads();
	pos[TIDY][TIDX] = TIDY * Code_Block_Size_X + TIDX;
	__syncthreads();
	// Set nbh
	unsigned int nbh = (!((cxds[TIDY][TIDX] & pass) && 1)) << POS_NEW_BITPOS;
	__syncthreads();
	// set pos_new and substrac 1 from pos if nbh > 0
	pos[TIDY][TIDX] = ((pos[TIDY][TIDX] & POS) - (nbh >> POS_NEW_BITPOS)) | nbh;
	__syncthreads();

	int warpVote = __any(nbh);
	__syncthreads();
	// voting across the blocks
	if((TID & (32 - 1)) == 0) atomicOr(blockVote, warpVote);
	__syncthreads();

	while(*blockVote) {
		// first thread of a block will reset the blockVote
		if(TID == 0) atomicAnd(blockVote, 0);
		__syncthreads();
		assert(*blockVote == 0);
		__syncthreads();
		warpVote = 0; // reset warpVote to zero
		// Get the predecessing neighbour pos state variable
		nbh = ((*(pos + TIDY * Code_Block_Size_X + TIDX - 1)) & POS_NEW);
		__syncthreads();

		pos[TIDY][TIDX] = pos[TIDY][TIDX] & (~POS_NEW);
		__syncthreads();
		// substrac 1 if nbh > 0 and jest pos_new
		pos[TIDY][TIDX] = (((pos[TIDY][TIDX] & POS) - (nbh >> POS_NEW_BITPOS)) & POS) | nbh;
		__syncthreads();
		// Voting
		warpVote = __any(nbh);
		__syncthreads();
		// execute it for the first thread of every warp only
		if((TID & (32 - 1)) == 0) atomicOr(blockVote, warpVote);
		__syncthreads();
	}
}

template <char Code_Block_Size_X>
__global__ void g_convert(CodeBlockAdditionalInfo *infos, unsigned int *g_icxds, unsigned char *g_ocxds, const int maxOutLength) {
	__shared__ unsigned int cxds[Code_Block_Size_X * Code_Block_Size_X];
//	__shared__ unsigned int pos[Code_Block_Size_X][Code_Block_Size_X];
	CodeBlockAdditionalInfo *info = &(infos[blockIdx.x]);
	int size = Code_Block_Size_X * Code_Block_Size_X;

//	if((TIDX >= info->width) || (TIDY >= info->height)) return;

	int curr_pos = 0;
	cxds[TIDY * Code_Block_Size_X + TIDX] = g_icxds[blockIdx.x * maxOutLength + TIDY * Code_Block_Size_X + TIDX];
	__syncthreads();
	if(TID == 0) {
		for(unsigned int h = 0; h < Code_Block_Size_X; ++h) {
			for(unsigned int w = 0; w < Code_Block_Size_X; ++w) {
				if(cxds[h * Code_Block_Size_X + w] & CUP) {
					unsigned char counter = cxds[h * Code_Block_Size_X + w] & CXD_COUNTER;
					for(unsigned char k = 0; k < counter; ++k) {
						unsigned char d = (cxds[h * Code_Block_Size_X + w] >> (D1_BITPOS - k * 6)) & 0x1;
						unsigned char cx = (cxds[h * Code_Block_Size_X + w] >> (CX1_BITPOS - k * 6)) & 0x1f;
//						printf("%d) %d %d\n", curr_pos, d, cx);
						g_ocxds[blockIdx.x * maxOutLength * 4 + curr_pos] = (d << 5) | cx;
						++curr_pos;
					}
				}
			}
		}
	}
	__syncthreads();

	for(unsigned char i = 1; i < info->significantBits; ++i) {
		cxds[TIDY * Code_Block_Size_X + TIDX] = g_icxds[blockIdx.x * maxOutLength + i * size + TIDY * Code_Block_Size_X + TIDX];
		__syncthreads();
		if(TID == 0) {
			for(unsigned char pass = SPP; pass < (CUP << 1); pass <<= 1) {
				for(unsigned int h = 0; h < Code_Block_Size_X; ++h) {
					for(unsigned int w = 0; w < Code_Block_Size_X; ++w) {
						if(cxds[h * Code_Block_Size_X + w] & pass) {
							unsigned char counter = cxds[h * Code_Block_Size_X + w] & CXD_COUNTER;
							for(unsigned char k = 0; k < counter; ++k) {
								unsigned char d = (cxds[h * Code_Block_Size_X + w] >> (D1_BITPOS - k * 6)) & 0x1;
								unsigned char cx = (cxds[h * Code_Block_Size_X + w] >> (CX1_BITPOS - k * 6)) & 0x1f;
//								printf("%d) %d %d\n", curr_pos, d, cx);
								g_ocxds[blockIdx.x * maxOutLength * 4 + curr_pos] = (d << 5) | cx;
								++curr_pos;
							}
						}
					}
				}
			}
		}
		__syncthreads();
	}
	if(TID == 0) {
		info->magconOffset = curr_pos;
	}
}

void convert(dim3 gridDim, dim3 blockDim, CodeBlockAdditionalInfo *infos, unsigned int *g_icxds, unsigned char *g_ocxds, const int maxOutLength)
{
	printf("dim %d %d\n", blockDim.x, blockDim.y);
	switch(blockDim.x) {
//	case 4: g_convert<4><<<gridDim, blockDim>>>(infos, g_icxds, g_ocxds, maxOutLength); break;
//	case 8: g_convert<8><<<gridDim, blockDim>>>(infos, g_icxds, g_ocxds, maxOutLength); break;
	case 16: g_convert<16><<<gridDim, blockDim>>>(infos, g_icxds, g_ocxds, maxOutLength); break;
	case 32: g_convert<32><<<gridDim, blockDim>>>(infos, g_icxds, g_ocxds, maxOutLength); break;
//	case 64: bpc_encoder<64><<<gridDim, blockDim>>>(infos, g_cxds); break;
	}

	cudaThreadSynchronize();
	cudaError_t cuerr;
	if (cuerr = cudaGetLastError()) {
		printf("bpc_encoder error: %s\n", cudaGetErrorString(cuerr));
		return;
	}
}
