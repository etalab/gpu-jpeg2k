/*
 * gpu_convert.cu
 *
 *  Created on: Feb 20, 2012
 *      Author: miloszc
 */

#include "gpu_bpc.h"

#define POS_NEW_BITPOS 30
#define POS_NEW (1 << POS_NEW_BITPOS)

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
	pos[TIDY][TIDX] = (!((cxds[TIDY][TIDX] & pass) && 1)) << POS_NEW_BITPOS;

	// Set nbh
	unsigned int nbh = ((*(pos + TIDY * Code_Block_Size_X + TIDX - 1)) & POS_NEW);
	__syncthreads();
	// set pos_new if nbh > 0
	pos[TIDY][TIDX] |= nbh;
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

		// We are interested in newly found positions only
		nbh &= ~(pos[TIDY][TIDX] & POS_NEW);
		__syncthreads();

		pos[TIDY][TIDX] |= nbh;
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
__global__ void convert(CodeBlockAdditionalInfo *infos, unsigned int *g_icxds, unsigned int *g_ocxds, const int maxOutLength) {
	__shared__ unsigned int cxds[Code_Block_Size_X][Code_Block_Size_X];
	__shared__ unsigned int pos[Code_Block_Size_X][Code_Block_Size_X];
	CodeBlockAdditionalInfo *info = &(infos[blockIdx.x]);
	int size = Code_Block_Size_X * Code_Block_Size_X;

	if(TID == 0) pos = 0;
	__syncthreads();

	for(unsigned char i = 1; i < info->significantBits; ++i) {
		cxds[TIDY][TIDX] = g_icxds[blockIdx.x * maxOutLength + i * size + TIDY * Code_Block_Size_X + TIDX];
		__syncthreads();
		for(unsigned char pass = SPP; pass < (CUP << 1); pass <<= 1) {
			unsigned char counter = cxds[TIDY][TIDX] & CXD_COUNTER;
			for(unsigned char k = 0; k < counter; ++k) {
				unsigned char d = (cxds[TIDY][TIDX] >> (D1_BITPOS - k * 6)) & 0x1;
				unsigned char cx = (cxds[TIDY][TIDX] >> (CX1_BITPOS - k * 6)) & 0x1f;
				if(cxds[TIDY][TIDX] & pass) {
					g_ocxds[blockIdx.x * maxOutLength + atomicAdd(&pos, 1)] = (d << 5) | cx;
				}
			}
		}
	}
}

void convert() {

}
