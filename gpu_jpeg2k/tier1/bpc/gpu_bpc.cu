/**
 * @file gpu_bpc.cu
 *
 * @author Milosz Ciznicki
 */
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "gpu_bpc.h"

__device__ void dbinary_printf(unsigned int in)
{
	for(int i = 0; i < 32; i++) {
		if((in >> (31 - i)) & 1)
			printf("1");
		else
			printf("0");
		if(i % 8 == 7)
			printf(" ");
	}

	printf("\n");
}

__device__ unsigned char getSPCX(unsigned int i, unsigned char subband)
{
	return SPCXLUT[subband][i & 0x1FF];
}

__device__ unsigned char getSICX(unsigned int sig_sign)
{
	return signcxlut[sig_sign];
}

template <char Code_Block_Size_X>
__device__ void save_cxd(unsigned int cxds[][Code_Block_Size_X], unsigned int &pairs, int add = 1) {
	// (x % (DIM /4)) * 4 + (y % 4)
	int bacx = (TIDX &((Code_Block_Size_X >> 2) - 1))*4 + (TIDY & 3);
	// floor(i /4) * 4 + floor(j /(DIM /4))
	int bacy = floorf(TIDY / 4)*4 + floorf(TIDX / (Code_Block_Size_X >> 2));
	pairs = (pairs & ~0x7) | ((pairs & 0x7) + add);
	cxds[bacy][bacx] = pairs;
//	if(cxds[bacy][bacx] == 0x5300022)
//		printf("%x %d %d	%d %d\n", cxds[bacy][bacx], bacy, bacx, TIDY, TIDX);
}

template <char Code_Block_Size_X>
__device__ void RunLengthCoding(unsigned int coeff[][Code_Block_Size_X + BORDER], unsigned int cxds[][Code_Block_Size_X], const unsigned char bitplane) {
	// store information about left and right neighbours
	coeff[Y][X] |= ((coeff[Y][X - 1] & SIGMA_NEW) ||
					(coeff[Y][X - 1] & SIGMA_OLD) ||
					(coeff[Y][X] & SIGMA_NEW) ||
					(coeff[Y][X] & SIGMA_OLD) ||
					(coeff[Y][X + 1] & SIGMA_NEW) ||
					(coeff[Y][X + 1] & SIGMA_OLD)) << RLC_BITPOS;
	__syncthreads();

	coeff[Y][X] |= (coeff[Y][X - 1] >> bitplane) << RLC_BITPOS;
	__syncthreads();

	// get RLC values of successors and save in first stripe position
	atomicOr(&coeff[((Y - BORDER) & 0xfffffffc) + BORDER][X], coeff[Y][X] & RLC);
	__syncthreads();

	// get 4 bits of current stripe
	int shift = 6 + (TIDY & 3);

	// >> bitplane & 1??
	atomicOr(&coeff[((Y - BORDER) & 0xfffffffc) + BORDER][X], (coeff[Y][X] & SIGMA_NEW) << shift);
	__syncthreads();

	// first thread in stripe
	if(!(TIDY & 3)) {
		// store information about top, bottom and corners
		unsigned int nbh = ((coeff[Y - 1][X - 1] & SIGMA_NEW) ||
				(coeff[Y - 1][X - 1] & SIGMA_OLD) || ((coeff[Y - 1][X - 1] >> bitplane) & 1) ||
				(coeff[Y - 1][X] & SIGMA_NEW) ||
				(coeff[Y - 1][X] & SIGMA_OLD) || ((coeff[Y - 1][X] >> bitplane) & 1) ||
				(coeff[Y - 1][X + 1] & SIGMA_NEW) ||
				(coeff[Y - 1][X + 1] & SIGMA_OLD) || ((coeff[Y - 1][X + 1] >> bitplane) & 1) ||
				(coeff[Y + 4][X - 1] & SIGMA_NEW) ||
				(coeff[Y + 4][X - 1] & SIGMA_OLD) || ((coeff[Y + 4][X - 1] >> bitplane) & 1) ||
				(coeff[Y + 4][X] & SIGMA_NEW) ||
				(coeff[Y + 4][X] & SIGMA_OLD) ||
				(coeff[Y + 4][X + 1] & SIGMA_NEW) ||
				(coeff[Y + 4][X + 1] & SIGMA_OLD)) << RLC_BITPOS;

		coeff[Y][X] |= nbh;
		// invert RLC bit
		coeff[Y][X] = (coeff[Y][X] & ~RLC) | (~coeff[Y][X] & RLC);
		//?
//		__syncthreads();
		// check if sigma_new, sigma_old are zero and rlc is one
		if((coeff[Y][X] & 0x1f) == 0x10) {
			unsigned int pairs = RLC_CX_17; // set CX =17
			// vector of stripe significance
			int cx17_d = (coeff[Y][X] >> 6) & 0xf;
			pairs |= ((cx17_d && 1) << D1_BITPOS); // set D
			if (cx17_d) {
				// if D is 1, then generate two CX 18
				int firstBitPos = __ffs(cx17_d) - 1;
				pairs |= RLC_CX_18;
				// code position of first 1 bit into D2 and D3
				pairs |= (firstBitPos & 0x2) << (D2_BITPOS -1);
				pairs |= (firstBitPos & 0x1) << D3_BITPOS;
				// mark bits processed by RLC in current bit-plane


				pairs |= 1 << CUP_BITPOS;
				// save CX,D paris
				cxds[TIDY][TIDX] = pairs;
			}
		}

	}
}

template <char Code_Block_Size_X>
__device__ void CleanUpPassMSB(unsigned int coeff[][Code_Block_Size_X + BORDER], unsigned int cxds[][Code_Block_Size_X], CodeBlockAdditionalInfo *info, const unsigned char bitplane) {
//	SET_SIGMA_NEW(coeff[Y][X], GET_SIGMA_NEW(coeff[Y][X], bitplane));
	coeff[Y][X] |= (coeff[Y][X] >> bitplane) & SIGMA_NEW;
	__syncthreads();

	// significance of left neighbors
	coeff[Y][X] |=  (coeff[Y][X - 1] & SIGMA_NEW) << RLC_BITPOS;
	__syncthreads();

	// get RLC values of successors and save in first stripe position
	atomicOr(&coeff[((Y - BORDER) & 0xfffffffc) + BORDER][X], coeff[Y][X] & RLC);
	__syncthreads();

	// get 4 bits of current stripe
	int shift = 6 + (TIDY & 3);

	atomicOr(&coeff[((Y - BORDER) & 0xfffffffc) + BORDER][X], (coeff[Y][X] & SIGMA_NEW) << shift);
	__syncthreads();

	unsigned int pairs = 0;

	// RLC
	if((TIDY & (4 - 1)) == 0) {
		coeff[Y][X] |=  ((coeff[Y - 1][X + 1] & SIGMA_NEW) |
						(coeff[Y - 1][X] & SIGMA_NEW) |
						(coeff[Y - 1][X - 1] & SIGMA_NEW)) << RLC_BITPOS;

		coeff[Y + 1][X] = coeff[Y + 1][X] & (~(1 << RLC_BITPOS));
		coeff[Y + 2][X] = coeff[Y + 2][X] & (~(1 << RLC_BITPOS));
		coeff[Y + 3][X] = coeff[Y + 3][X] & (~(1 << RLC_BITPOS));
		// invert RLC bit
		coeff[Y][X] = (coeff[Y][X] & ~RLC) | (~coeff[Y][X] & RLC);

//		if((TIDY == 4) && (TIDX == 1)) {
//			printf("RLC %x %d %d	%d\n", coeff[Y][X], TIDY, TIDX, bitplane);
//			dbinary_printf(coeff[Y][X]);
//			dbinary_printf(coeff[Y - 1][X]);
//		}
		if(coeff[Y][X] & RLC) {
			pairs = RLC_CX_17; // set CX =17
			// vector of stripe significance
			int cx17_d = (coeff[Y][X] >> 6) & 0xf;
			pairs |= ((cx17_d && 1) << D1_BITPOS); // set D
//			coeff[Y + 0][X] |= 1 << RLC_BITPOS;
			coeff[Y + 1][X] |= 1 << RLC_BITPOS;
			coeff[Y + 2][X] |= 1 << RLC_BITPOS;
			coeff[Y + 3][X] |= 1 << RLC_BITPOS;
			if (cx17_d) {
				// if D is 1, then generate two CX 18
				int firstBitPos = __ffs(cx17_d) - 1;
				pairs |= RLC_CX_18;
				// code position of first 1 bit into D2 and D3
				pairs |= (firstBitPos & 0x2) << (D2_BITPOS -1);
				pairs |= (firstBitPos & 0x1) << D3_BITPOS;
				// mark bits processed by RLC in current bit-plane
				// first in stripe is always encoded
				coeff[Y + 1][X] = coeff[Y + 1][X] & ~(!((firstBitPos & 0x1) | (firstBitPos & 0x2)) << RLC_BITPOS);
				coeff[Y + 2][X] = coeff[Y + 2][X] & ~(!(firstBitPos & 0x2) << RLC_BITPOS);
				coeff[Y + 3][X] = coeff[Y + 3][X] & ~(!((firstBitPos & 0x1) && (firstBitPos & 0x2)) << RLC_BITPOS);
//				printf("RLC %x\n", pairs);
//				cxds[TIDY][TIDX] = pairs;
			}
			// save CX,D paris 1 or 3
			save_cxd(cxds, pairs, (cx17_d && 1) - (cx17_d == 0) + 2);
		}
	}
	__syncthreads();

//	printf("%x\n", coeff[Y][X]);
	// ZC
	if((coeff[Y][X] & RLC) == 0) {
		// in stripe significance
		// 00	01,10	11 - stripe position
		// ooo	oo		oo
		// ox	ox		ox
		// o	o
		unsigned int sig = (((coeff[Y - 1][X + 1] & SIGMA_NEW) && ((TIDY & 3) == 0x0)) << 2) |
							((coeff[Y - 1][X] & SIGMA_NEW) << 1) |
							((coeff[Y - 1][X - 1] & SIGMA_NEW) << 0) |
							((coeff[Y][X - 1] & SIGMA_NEW) << 3) |
							(((coeff[Y + 1][X - 1] & SIGMA_NEW) && !((TIDY & 3) == 0x3)) << 6);

		pairs = getSPCX(sig, info->subband) << CX1_BITPOS; // set CX
		pairs |= (coeff[Y][X] & SIGMA_NEW) << D1_BITPOS; // set D

		pairs |= (1 << CUP_BITPOS);
//		if((TIDY == 4) && (TIDX == 1))
//			printf("ZC %x %d %d\n", coeff[Y][X], TIDY, TIDX);
		save_cxd(cxds, pairs);
//		cxds[TIDY][TIDX] = pairs;
	}
	__syncthreads();

	// SC
	if(coeff[Y][X] & SIGMA_NEW) {
		unsigned int sig_sign = ((coeff[Y - 1][X] & SIGMA_NEW) << 7)/*V0*/ |
							((coeff[Y - 1][X] >> SIGN_BITPOS) << 6)/*V0*/ |
							((coeff[Y ][X - 1] & SIGMA_NEW) << 5)/*H0*/ |
							((coeff[Y][X - 1] >> SIGN_BITPOS) << 4)/*H0*/;
		unsigned char cx_x = getSICX(sig_sign);
		unsigned char cx = cx_x & 0xF; // set CX
		unsigned char d = (coeff[Y][X] >> SIGN_BITPOS) ^ ((cx_x >> 4) & 1); // set D
		// shift by 6 (ZC), 18 (RLC on 1st bit), 0 (RLC on 2nd...4th bit)
		int shift = ((!(coeff[Y][X] & RLC)) * 6) + ((coeff[Y][X] & RLC) && (!(TIDY & 3))) * 18;
		pairs |= (cx << (D1_BITPOS + 1 - shift)); // save CX
		pairs |= (d << (D1_BITPOS - shift)); // save D
//		if((TIDY == 4) && (TIDX == 1))
//			printf("SC %x %d %d\n", coeff[Y][X], TIDY, TIDX);
		save_cxd(cxds, pairs);
//		cxds[TIDY][TIDX] = pairs;
	}
}

template <char Code_Block_Size_X>
__global__ void bpc_encoder(CodeBlockAdditionalInfo *infos, unsigned int *g_cxds) {
	// to access coeff use X and Y
	__shared__ unsigned int coeff[Code_Block_Size_X + BORDER][Code_Block_Size_X + BORDER];
	__shared__ unsigned int cxds[Code_Block_Size_X][Code_Block_Size_X];
	__shared__ unsigned int maxs[Code_Block_Size_X];

	CodeBlockAdditionalInfo *info = &(infos[blockIdx.x]);

	if((TIDX >= info->width) || (TIDY >= info->height)) return;

	int cache_value = info->coefficients[TIDY * info->nominalWidth + TIDX];
	coeff[Y][X] = cache_value/* < 0 ? (1 << 31) | (-cache_value) : cache_value*/;
//	if(TIDY == 0)
//		printf("%x\n", coeff[Y][X]);
	__syncthreads();

	// find most significant bitplane
	unsigned int tmp = 0;
	if((TIDX < Code_Block_Size_X) && (TIDY == 0)) {
		for(int i = BORDER; i < Code_Block_Size_X + BORDER; ++i) {
			tmp = max(tmp, coeff[X][i] & MAGBITS);
		}
		maxs[TIDX] = tmp;
	}
	__syncthreads();

	if((TIDX == 0) && (TIDY == 0)) {
		tmp = 0;
		for(int i = 0; i < Code_Block_Size_X; ++i) {
			tmp = max(tmp, maxs[i]);
		}
		maxs[0] = 31 - __clz(tmp);
	}
	__syncthreads();

	unsigned char leastSignificantBP = 31 - info->magbits;
	unsigned char significantBits = maxs[0] - leastSignificantBP + 1;

	if(significantBits == 0) return;

//	printf("%d\n", leastSignificantBP + significantBits - 1);

	CleanUpPassMSB<Code_Block_Size_X>(coeff, cxds, info, leastSignificantBP + significantBits - 1);
	__syncthreads();

	// (x % (DIM /4)) * 4 + (y % 4)
	int bacx = (TIDX &((Code_Block_Size_X >> 2) - 1))*4 + (TIDY & 3);
	// floor(i /4) * 4 + floor(j /(DIM /4))
	int bacy = floorf(TIDY / 4)*4 + floorf(TIDX / (Code_Block_Size_X >> 2));
//	if(bacy * info->height + bacx == 132)
//		printf("%x %d %d\n", cxds[bacy][bacx], TIDY, TIDX);
	g_cxds[bacy * info->height + bacx] = cxds[bacy][bacx];
}

void launch_bpc_encode(dim3 gridDim, dim3 blockDim, CodeBlockAdditionalInfo *infos, unsigned int *g_cxds)
{
	switch(blockDim.x) {
	case 4: bpc_encoder<4><<<gridDim, blockDim>>>(infos, g_cxds); break;
	case 8: bpc_encoder<8><<<gridDim, blockDim>>>(infos, g_cxds); break;
	case 16: bpc_encoder<16><<<gridDim, blockDim>>>(infos, g_cxds); break;
	case 32: bpc_encoder<32><<<gridDim, blockDim>>>(infos, g_cxds); break;
	case 64: bpc_encoder<64><<<gridDim, blockDim>>>(infos, g_cxds); break;
	}

	cudaThreadSynchronize();
	cudaError_t cuerr;
	if (cuerr = cudaGetLastError()) {
		printf("bpc_encoder error: %s\n", cudaGetErrorString(cuerr));
		return;
	}
}
