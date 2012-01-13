/**
 * @file gpu_bpc.cu
 *
 * @author Milosz Ciznicki
 */
#include <stdio.h>
#include "gpu_bpc.h"

#define MAGBITS 0x7FFFF000
#define BORDER 2
#define SIGMA_NEW 1
#define SIGMA_OLD 2

// CX,D pairs vector
//	31-25	26	25-21		20	19-15		14		13-9	8		5		4		3		2-0
//	CX1	|	D1	|	CX2	|	D2	|	CX3	|	D3	|	CX4	|	D4	|	CUP	|	MRP	|	SRP	|	CX,D pairs counter
#define CX1_BITPOS 25
// coeff
//	31		30-10		9-6						5		4			2		1				0
//	Sign	|	pixel	|	rlcDposition	|	coded	|	rlc	|	nbh	|	sigma_old	|	sigma_new
// 4 - RLC
#define RLC 0x00000010
#define CODED 0x00000020

#define SIGN_BITPOS 31
#define SIGN 1 << SIGN_BITPOS

#define X (threadIdx.x + BORDER)
#define Y (threadIdx.y + BORDER)

// 31-27 position set CX 17
#define RLC_CX_17 0x88000000
// 25-21 and 19-15 position set CX18
#define RLC_CX_18 0x02490000

#define D1_BITPOS 26
#define D2_BITPOS 20
#define D3_BITPOS 14
#define D4_BITPOS 8

#define RLC_BITPOS 4
#define CODED_BITPOS 5
#define GET_SIGMA_NEW(src, bitplane) ((src >> bitplane) & SIGMA_NEW)
#define SET_SIGMA_NEW(dst, src) (dst |= src)
#define GET_VAR(src, var) (src & var)

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

__device__ void RunLengthCoding(unsigned int **coeff, unsigned int **cxds, const unsigned char bitplane) {
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
	atomicOr(&coeff[((Y/* - 1*/) & 0xfffffffc)/* + 1*/][X], coeff[Y][X] & RLC);
	__syncthreads();

	// get 4 bits of current stripe
	int shift = 6 + (Y & 3);

	// >> bitplane & 1??
	atomicOr(&coeff[Y & 0xfffffffc][X], (coeff[Y][X] & SIGMA_NEW) << shift);
	__syncthreads();

	// first thread in stripe
	if(!(Y & 3)) {
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

				// save CX,D paris
				cxds[Y][X] = pairs;
			}
		}

	}
}

template <char Code_Block_Size_X>
__device__ void CleanUpPassMSB(unsigned int coeff[][Code_Block_Size_X], unsigned int cxds[][Code_Block_Size_X], CodeBlockAdditionalInfo *info, const unsigned char bitplane) {
	SET_SIGMA_NEW(coeff[Y][X], GET_SIGMA_NEW(coeff[Y][X], bitplane));
	__syncthreads();

	// significance of left neighbors
	coeff[Y][X] |=  (coeff[Y][X - 1] & SIGMA_NEW) << RLC_BITPOS;
	__syncthreads();

	// get RLC values of successors and save in first stripe position
	atomicOr(&coeff[((Y/* - 1*/) & 0xfffffffc)/* + 1*/][X], coeff[Y][X] & RLC);
	__syncthreads();

//	unsigned char rlcNbh = (coeff[Y - 1][X] & SIGMA_NEW) |
//	(coeff[Y - 1][X -1] & SIGMA_NEW) |
//	(coeff[Y][X -1] & SIGMA_NEW) |
//	(coeff[Y + 1][X -1] & SIGMA_NEW) |
//	(coeff[Y + 2][X -1] & SIGMA_NEW) |
//	(coeff[Y + 3][X -1] & SIGMA_NEW);
//	__syncthreads();

	// get 4 bits of current stripe
	int shift = 6 + (Y & 3);

	atomicOr(&coeff[Y & 0xfffffffc][X], (coeff[Y][X] & SIGMA_NEW) << shift);
	__syncthreads();

	// RLC
	if((Y & (4 - 1)) == 0) {
		coeff[Y][X] |=  ((coeff[Y - 1][X] & SIGMA_NEW) |
						(coeff[Y - 1][X - 1] & SIGMA_NEW)) << RLC_BITPOS;
		// invert RLC bit
		coeff[Y][X] = (coeff[Y][X] & ~RLC) | (~coeff[Y][X] & RLC);
		if((coeff[Y][X] & RLC) == 1) {
			unsigned int pairs = RLC_CX_17; // set CX =17
			// vector of stripe significance
			int cx17_d = (coeff[Y][X] >> 6) & 0xf;
			pairs |= ((cx17_d && 1) << D1_BITPOS); // set D
			coeff[Y + 0][X] |= 1 << CODED_BITPOS;
			coeff[Y + 1][X] |= 1 << CODED_BITPOS;
			coeff[Y + 2][X] |= 1 << CODED_BITPOS;
			coeff[Y + 3][X] |= 1 << CODED_BITPOS;
			if (cx17_d) {
				// if D is 1, then generate two CX 18
				int firstBitPos = __ffs(cx17_d) - 1;
				pairs |= RLC_CX_18;
				// code position of first 1 bit into D2 and D3
				pairs |= (firstBitPos & 0x2) << (D2_BITPOS -1);
				pairs |= (firstBitPos & 0x1) << D3_BITPOS;
				// mark bits processed by RLC in current bit-plane
				// first in stripe is always encoded
				coeff[Y + 1][X] |= ((firstBitPos & 0x1) | (firstBitPos & 0x2)) << CODED_BITPOS;
				coeff[Y + 2][X] |= (firstBitPos & 0x2) << CODED_BITPOS;
				coeff[Y + 3][X] |= ((firstBitPos & 0x1) & (firstBitPos & 0x2)) << CODED_BITPOS;
				// save CX,D paris
				cxds[Y][X] = pairs;
			}
		}
	}
	__syncthreads();

	// ZC
	if((coeff[Y][X] & CODED) == 0) {
		// in stripe significance
		// 00	01,10	11 - stripe position
		// ooo	oo		oo
		// ox	ox		ox
		// o	o
		unsigned int sig = (((coeff[Y - 1][X + 1] & SIGMA_NEW) && ((Y & 3) == 0x0)) << 2) |
							((coeff[Y - 1][X] & SIGMA_NEW) << 1) |
							((coeff[Y - 1][X - 1] & SIGMA_NEW) << 0) |
							((coeff[Y][X - 1] & SIGMA_NEW) << 3) |
							(((coeff[Y + 1][X - 1] & SIGMA_NEW) && !((Y & 3) == 0x3)) << 6);

		unsigned int pairs = getSPCX(sig, info->subband) << CX1_BITPOS; // set CX
		pairs |= (coeff[Y][X] & SIGMA_NEW) << D1_BITPOS; // set D
		cxds[Y][X] = pairs;
	}

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
		int shift = ((!(coeff[Y][X] & RLC)) * 6) + (((coeff[Y][X] & RLC)) && (!(Y & 3)) * 18);
		unsigned int pairs = (cx << (D1_BITPOS + 1 - shift)); // save CX
		pairs |= (d << (D1_BITPOS - shift)); // save D
		cxds[Y][X] = pairs;
	}
}

template <char Code_Block_Size_X>
__global__ void bpc_encoder(CodeBlockAdditionalInfo *infos) {
	__shared__ unsigned int coeff[Code_Block_Size_X][Code_Block_Size_X];
	__shared__ unsigned int cxds[Code_Block_Size_X][Code_Block_Size_X];
	__shared__ unsigned int maxs[Code_Block_Size_X];

	CodeBlockAdditionalInfo *info = &(infos[blockIdx.x]);

	int cache_value = info->coefficients[Y * info->nominalWidth + X];
	coeff[Y][X] = cache_value < 0 ? (1 << 31) | (-cache_value) : cache_value;
	__syncthreads();

	// find most significant bitplane
	unsigned int tmp = 0;
	if((X < Code_Block_Size_X) && (Y == 0)) {
		for(int i = 0; i < Code_Block_Size_X; ++i) {
			tmp = max(tmp, coeff[X][i] & MAGBITS);
		}
		maxs[X] = tmp;
	}
	__syncthreads();

	if((X == 0) && (Y == 0)) {
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

	CleanUpPassMSB<Code_Block_Size_X>(coeff, cxds, info, leastSignificantBP + significantBits - 1);
}

void launch_bpc_encode(dim3 gridDim, dim3 blockDim, CodeBlockAdditionalInfo *infos)
{
	bpc_encoder<16 + BORDER><<<gridDim, blockDim>>>(infos);
}
