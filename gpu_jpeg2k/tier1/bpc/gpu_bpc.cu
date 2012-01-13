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
#define RLC 4

#define X threadIdx.x
#define Y threadIdx.y

#define RLC_BITPOS (RLC - 1)
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

__device__ void RunLengthCoding(unsigned int *coeff, const unsigned char bitplane) {
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

	atomicOr(&coeff[Y & 0xfffffffc][X], (coeff[Y][X] & SIGMA_NEW) << shift);
	__syncthreads();

	// first thread in stripe
	if(!(Y & 3)) {
		// store information about top, bottom and corners
		unsigned int nbh = ((coeff[Y - 1][X - 1] & SIGMA_NEW) ||
				(coeff[Y - 1][X - 1] & SIGMA_OLD) || ((coeff[Y - 1][X - 1] >> btiplane) & 1) ||
				(coeff[Y - 1][X] & SIGMA_NEW) ||
				(coeff[Y - 1][X] & SIGMA_OLD) || ((coeff[Y - 1][X] >> btiplane) & 1) ||
				(coeff[Y - 1][X + 1] & SIGMA_NEW) ||
				(coeff[Y - 1][X + 1] & SIGMA_OLD) || ((coeff[Y - 1][X + 1] >> btiplane) & 1) ||
				(coeff[Y + 4][X - 1] & SIGMA_NEW) ||
				(coeff[Y + 4][X - 1] & SIGMA_OLD) || ((coeff[Y + 4][X - 1] >> btiplane) & 1) ||
				(coeff[Y + 4][X] & SIGMA_NEW) ||
				(coeff[Y + 4][X] & SIGMA_OLD) ||
				(coeff[Y + 4][X + 1] & SIGMA_NEW) ||
				(coeff[Y + 4][X + 1] & SIGMA_OLD)) << RLC_BITPOS;

		coeff[Y][X] |= nbh;

		coeff[Y][X] = (coeff[Y][X] & ~RLC) | (~coeff[Y][X] & RLC);
		__syncthreads();

		if(( state [ i ][ j ]&0 xf ) == 16) {
		pairs = RLC_CX_17 ; // set CX =17
		int cx17_d = ( state [ i ][ j ] > >6) & 0 xf ;
		pairs |= (( cx17_d && 1) << D1_BITPOS ); // set D
		if ( cx17_d ) {
		// if D is 1 , then ge n er a te two CX 18
		int firstBitPos = __ffs ( cx17_d ) - 1;
		pairs |= RLC_CX_18 ;
		// code po s it i on of first 1 bit into D2 and D3
		pairs |= ( firstBitPos & 0 x2 ) < < ( D2_BITPOS -1);
		pairs |= ( firstBitPos & 0 x1 ) < < D3_BITPOS ;
		/* mark bits p r o c e s s e d by RLC in current bit - plane */
		. . .
		}
		}

	}
}

__device__ void CleanUpPassMSB(unsigned int *coeff, const unsigned char bitplane) {
	SET_SIGMA_NEW(coeff[Y][X], GET_SIGMA_NEW(coeff[Y][X], bitplane));
	__syncthreads();

	unsigned char rlcNbh = ((coeff[Y - 1][X] >> bitplane) & SIGMA_NEW) |
	((coeff[Y - 1][X -1] >> bitplane) & SIGMA_NEW) |
	((coeff[Y][X -1] >> bitplane) & SIGMA_NEW) |
	((coeff[Y + 1][X -1] >> bitplane) & SIGMA_NEW) |
	((coeff[Y + 2][X -1] >> bitplane) & SIGMA_NEW) |
	((coeff[Y + 3][X -1] >> bitplane) & SIGMA_NEW);

	if((rlcNbh == 0) && ((Y & (4 - 1)) == 0)) {
		// RLC
	}

	if(rlcNbh == 1) {
		// ZC
	}
	if(GET_VAR(coeff[Y][X], SIGMA_NEW) == 1) {
		// SC
	}
}

template <char Code_Block_Size_X>
__global__ void bpc_encoder(CodeBlockAdditionalInfo *infos) {
	__shared__ unsigned int coeff[Code_Block_Size_X + BORDER][Code_Block_Size_X + BORDER];
	__shared__ unsigned int cxds[Code_Block_Size_X][Code_Block_Size_X];
	__shared__ unsigned int maxs[Code_Block_Size_X];

	CodeBlockAdditionalInfo *info = &(infos[blockIdx.x]);

	unsigned int cache_value = info->coefficients[Y * info->nominalWidth + X];
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
	unsigned char significantBits = max[0] - leastSignificantBP + 1;

	if(significantBits =< 0) return;


}

void launch_bpc_encode(dim3 gridDim, dim3 blockDim, CodeBlockAdditionalInfo *infos)
{
	bpc_encoder<16><<<gridDim, blockDim>>>(infos);
}
