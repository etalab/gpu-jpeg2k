/**
 * @file gpu_bpc.cu
 *
 * @author Milosz Ciznicki
 */
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <assert.h>
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
//	if(cxds[bacy][bacx] == 0x3d30002a)
//		printf("%x %d %d	%d %d\n", cxds[bacy][bacx], bacy, bacx, TIDY, TIDX);
}

template <char Code_Block_Size_X>
__device__ void cleanUpPassMSB(unsigned int coeff[][Code_Block_Size_X + 2*BORDER], unsigned int cxds[][Code_Block_Size_X], CodeBlockAdditionalInfo *info, const unsigned char bitplane) {
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
			pairs |= (1 << CUP_BITPOS);
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
//		if((TIDY == 1) && (TIDX == 0))
//			printf("ZC %x %d %d\n", coeff[Y][X], TIDY, TIDX);
		save_cxd(cxds, pairs);
//		cxds[TIDY][TIDX] = pairs;
	}
	__syncthreads();

	// SC
	if(coeff[Y][X] & SIGMA_NEW) {
		unsigned int sig_sign = ((coeff[Y - 1][X] & SIGMA_NEW) << 7)/*V0*/ |
							((coeff[Y - 1][X] >> SIGN_BITPOS) << 6)/*V0*/ |
							((coeff[Y][X - 1] & SIGMA_NEW) << 5)/*H0*/ |
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
		pairs |= (1 << CUP_BITPOS);
		save_cxd(cxds, pairs);
//		cxds[TIDY][TIDX] = pairs;
	}
}

template <char Code_Block_Size_X>
__device__ void btiplanePreprocessing(unsigned int coeff[][Code_Block_Size_X + 2*BORDER], int &blockVote, const unsigned char bitplane) {
	// Set sigma_old
	coeff[Y][X] |= (coeff[Y][X] & SIGMA_NEW) << 1;
	// Unset sigma_new
	//coeff[Y][X] &= ~SIGMA_NEW;
	coeff[Y][X] &= ~0x3f5/*CLR_VAR*/;
	__syncthreads();

	// Set nbh
	unsigned char nbh = ((coeff[Y - 1][X + 1] & SIGMA_OLD) ||
					(coeff[Y - 1][X] & SIGMA_OLD) ||
					(coeff[Y - 1][X - 1] & SIGMA_OLD) ||
					(coeff[Y][X - 1] & SIGMA_OLD) ||
					(coeff[Y + 1][X - 1] & SIGMA_OLD) ||
					(coeff[Y + 1][X] & SIGMA_OLD) ||
					(coeff[Y + 1][X + 1] & SIGMA_OLD) ||
					(coeff[Y][X + 1] & SIGMA_OLD));
	__syncthreads();
	// sigma_old == 0 and nbh == 1 and bit value == 1 set sigma_new
	coeff[Y][X] |= ((!((coeff[Y][X] & SIGMA_OLD) >> 1)) & nbh & ((coeff[Y][X] >> bitplane) & 1));
	nbh &= ((!((coeff[Y][X] & SIGMA_OLD) >> 1)) & ((coeff[Y][X] >> bitplane) & 1));
	__syncthreads();

	int warpVote = __any(nbh);
	__syncthreads();
	// voting across the blocks
	if((TID & (32 - 1)) == 0) atomicOr(&blockVote, warpVote);
	__syncthreads();

	while(blockVote) {
		// first thread of a block will reset the blockVote
		if(TID == 0) atomicAnd(&blockVote, 0);
		__syncthreads();
		warpVote = 0; // reset warpVote to zero
		// Get the predecessing neighbour significance state variables
		nbh = (((coeff[Y - 1][X + 1] & SIGMA_NEW) & ((TIDY & 3) == 0x0)) |
				(coeff[Y - 1][X] & SIGMA_NEW) |
				(coeff[Y - 1][X - 1] & SIGMA_NEW) |
				(coeff[Y][X - 1] & SIGMA_NEW) |
				((coeff[Y + 1][X - 1] & SIGMA_NEW) & ((TIDY & 3) != 0x3)));
		__syncthreads();
	//	. . .
		// IF nbh == 1 && bp [ x ][ y ]== 1 && S I G M A _ O L D == 0
		// THEN set SIGMA_NEW = 1
//		coeff[Y][X] |= ((!((coeff[Y][X] & SIGMA_OLD) >> 1)) & nbh & ((coeff[Y][X] >> bitplane) & 1));
		nbh &= ((!((coeff[Y][X] & SIGMA_OLD) >> 1)) & ((coeff[Y][X] >> bitplane) & 1));
		// We are interested in newly found sigmas only
		nbh &= ~(coeff[Y][X] & SIGMA_NEW);
		__syncthreads();
	//	. . .
		// IF nbh == 1 && bp [ x ][ y ]== 1 && S I G M A _ O L D == 0
                // THEN set SIGMA_NEW = 1
                coeff[Y][X] |= ((!((coeff[Y][X] & SIGMA_OLD) >> 1)) & nbh & ((coeff[Y][X] >> bitplane) & 1));
		// Voting
		warpVote = __any(nbh);
		__syncthreads();
		// execute it for the first thread of every warp only
		if((TID & (32 - 1)) == 0) atomicOr(&blockVote, warpVote);
		__syncthreads();
	}
}

template <char Code_Block_Size_X>
__device__ void magnitudeRefinementCoding(const unsigned int coeff[][Code_Block_Size_X + 2*BORDER], unsigned int cxds[][Code_Block_Size_X], const unsigned char bitplane) {
	if(coeff[Y][X] & SIGMA_OLD) {
		// in stripe significance - sigma_new
		// 00	01,10	11 - stripe position
		// ooo	oo		oo
		// ox	ox		ox
		// o	o
		unsigned char sig =	(((coeff[Y - 1][X + 1] >> bitplane) & 1) & ((TIDY & 3) == 0x0)) | /*tr*/
					((coeff[Y - 1][X] & SIGMA_NEW) | (coeff[Y - 1][X] & SIGMA_OLD) | ((coeff[Y - 1][X] >> bitplane) & 1)) | /*tc*/
					((coeff[Y - 1][X - 1] & SIGMA_NEW) | (coeff[Y - 1][X - 1] & SIGMA_OLD) | (((coeff[Y - 1][X - 1] >> bitplane) & 1))) | /*tl*/
					((coeff[Y][X - 1] & SIGMA_NEW) | (coeff[Y][X - 1] & SIGMA_OLD) | ((coeff[Y][X - 1] >> bitplane) & 1)) | /*l*/
					(((coeff[Y + 1][X - 1] >> bitplane) & 1) & ((TIDY & 3) != 0x3)); /*bl*/

		// in stripe significance - sigma_old
		// 11	01,10	00 - stripe position
		// 	 o	 o
		//  xo	xo		xo
		// ooo	oo		oo
		sig |= ((coeff[Y - 1][X + 1] & SIGMA_OLD) | (coeff[Y - 1][X + 1] & SIGMA_NEW)) | /*tr*/
			((coeff[Y][X + 1] & SIGMA_OLD) | (coeff[Y][X + 1] & SIGMA_NEW)) | /*r*/
			((coeff[Y + 1][X + 1] & SIGMA_OLD) | (coeff[Y + 1][X + 1] & SIGMA_NEW)) | /*br*/
			((coeff[Y - 1][X] & SIGMA_OLD) | (coeff[Y - 1][X] & SIGMA_NEW)) | /*bc*/
			((coeff[Y + 1][X - 1] & SIGMA_OLD) | (coeff[Y + 1][X - 1] & SIGMA_NEW)); /*bl*/

		unsigned char sigma_prim = (31 - __clz(coeff[Y][X] & 0x7fffffff) - bitplane) > 1;
		// if sig_prim == 0 and sig > 0 set CX 15, else set CX 14
		unsigned int pairs = (MRC_CX_14 | ((sig > 0) | (sigma_prim & 1)));//set CX
		// if sig_prim == 1 set CX 16
		pairs = (pairs << ((sigma_prim & 1) << 2)) << CX1_BITPOS;
		pairs |= ((coeff[Y][X] >> bitplane) & 1) << D1_BITPOS; // set D
		pairs |= 1 << MRP_BITPOS;
//		if((TIDY == 15) && (TIDX == 12) && (bitplane == 29))
//                        printf("MRC %d %d %x\n", TIDY, TIDX, pairs);
		save_cxd(cxds, pairs);
	}
}

template <char Code_Block_Size_X>
__device__ void runLengthCoding(unsigned int coeff[][Code_Block_Size_X + 2*BORDER], unsigned int cxds[][Code_Block_Size_X], unsigned int &pairs, const unsigned char bitplane) {
	// store information about current, left and right neighbours
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
	
	// clear stripe
	coeff[Y][X] &= ~0x3c0;
	__syncthreads();
//      printf("%d\n", leastSignificantBP + significantBits - 1);
//	if((TIDY == 4) && (TIDX == 4) && (bitplane == 29))
//        	printf("RLC1 %d %d %x    %d\n", TIDY, TIDX, (coeff[Y][X] >> 6) & 0xf, (coeff[Y][X] >> bitplane) & 1);

	// get 4 bits of current stripe
	int shift = 6 + (TIDY & 3);

	atomicOr(&coeff[((Y - BORDER) & 0xfffffffc) + BORDER][X], ((coeff[Y][X] >> bitplane) & 1) << shift);
	__syncthreads();

//	if((TIDY == 4) && (TIDX == 4) && (bitplane == 29))
//        	printf("RLC2 %d %d %x    %d\n", TIDY, TIDX, (coeff[Y][X] >> 6) & 0xf, (coeff[Y][X] >> bitplane) & 1);

	// first thread in stripe
	if(!(TIDY & 3)) {
		// store information about top, bottom and corners
		coeff[Y][X] |= ((coeff[Y - 1][X - 1] & SIGMA_NEW) ||
				(coeff[Y - 1][X - 1] & SIGMA_OLD) || ((coeff[Y - 1][X - 1] >> bitplane) & 1) ||
				(coeff[Y - 1][X] & SIGMA_NEW) ||
				(coeff[Y - 1][X] & SIGMA_OLD) || ((coeff[Y - 1][X] >> bitplane) & 1) ||
				(coeff[Y - 1][X + 1] & SIGMA_NEW) ||
				(coeff[Y - 1][X + 1] & SIGMA_OLD) || ((coeff[Y - 1][X + 1] >> bitplane) & 1) ||
				(coeff[Y + 4][X - 1] & SIGMA_NEW) ||
				(coeff[Y + 4][X - 1] & SIGMA_OLD) /*|| ((coeff[Y + 4][X - 1] >> bitplane) & 1)*/ ||
				(coeff[Y + 4][X] & SIGMA_NEW) ||
				(coeff[Y + 4][X] & SIGMA_OLD) ||
				(coeff[Y + 4][X + 1] & SIGMA_NEW) ||
				(coeff[Y + 4][X + 1] & SIGMA_OLD)) << RLC_BITPOS;

//		coeff[Y][X] |= nbh;
		coeff[Y + 1][X] = coeff[Y + 1][X] & (~(1 << RLC_BITPOS));
		coeff[Y + 2][X] = coeff[Y + 2][X] & (~(1 << RLC_BITPOS));
		coeff[Y + 3][X] = coeff[Y + 3][X] & (~(1 << RLC_BITPOS));
		// invert RLC bit
		coeff[Y][X] = (coeff[Y][X] & ~RLC) | (~coeff[Y][X] & RLC);
		//?
//		__syncthreads();
		// check if sigma_new, sigma_old are zero and rlc is one
		if((coeff[Y][X] & 0x1f) == 0x10) {
//			if((TIDY == 4) && (TIDX == 4) && (bitplane == 29))
//	                        printf("RLC %d %d %x	%d\n", TIDY, TIDX, (coeff[Y][X] >> 6) & 0xf, (coeff[Y][X] >> bitplane) & 1);

			pairs = RLC_CX_17; // set CX =17
			// vector of stripe significance
			int cx17_d = (coeff[Y][X] >> 6) & 0xf;
			pairs |= ((cx17_d && 1) << D1_BITPOS); // set D
			pairs |= 1 << CUP_BITPOS;
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
			}
			// save CX,D paris 1 or 3
			save_cxd(cxds, pairs, (cx17_d && 1) - (cx17_d == 0) + 2);
		}
	}
}

template <char Code_Block_Size_X>
__device__ void zeroCoding(CodeBlockAdditionalInfo *info, unsigned int coeff[][Code_Block_Size_X + 2*BORDER], unsigned int cxds[][Code_Block_Size_X], unsigned int &pairs, const unsigned char bitplane) {
	if(((coeff[Y][X] & RLC) == 0) && ((coeff[Y][X] & SIGMA_OLD) == 0)) {
		// in stripe significance - sigma_new
		// 00	01,10	11 - stripe position
		// ooo	oo		oo
		// ox	ox		ox
		// o	o						// sigma_old ?
		unsigned int nbh =	(((coeff[Y - 1][X + 1] & SIGMA_NEW) && ((TIDY & 3) == 0x0)) << 2) | /*tr*/
					(((coeff[Y - 1][X] & SIGMA_NEW) | ((coeff[Y - 1][X] & SIGMA_OLD) >> 1)) << 1) | /*tc*/
					(((coeff[Y - 1][X - 1] & SIGMA_NEW) | ((coeff[Y - 1][X - 1] & SIGMA_OLD) >> 1)) << 0) | /*tl*/
					(((coeff[Y][X - 1] & SIGMA_NEW) | ((coeff[Y][X - 1] & SIGMA_OLD) >> 1)) << 3) | /*l*/
					(((coeff[Y + 1][X - 1] & SIGMA_NEW) && ((TIDY & 3) != 0x3)) << 6); /*bl*/

		// in stripe significance - sigma_old
		// 11	01,10	00 - stripe position
		// 	 o	 o
		//  xo	xo		xo
		// ooo	oo		oo
		nbh |= (((coeff[Y - 1][X + 1] & SIGMA_OLD) >> 1) << 2) | /*tr*/
			(((coeff[Y][X + 1] & SIGMA_OLD) >> 1) << 5) | /*r*/
			(((coeff[Y + 1][X + 1] & SIGMA_OLD) >> 1) << 8) | /*br*/
			(((coeff[Y + 1][X] & SIGMA_OLD) >> 1) << 7) | /*bc*/
			(((coeff[Y + 1][X - 1] & SIGMA_OLD) >> 1) << 6); /*bl*/

		pairs = ((nbh == 0) << CUP_BITPOS) | ((nbh && 1) << SPP_BITPOS); // set CUP or SPP, nbh differentiate

//		if((TIDY == 3) && (TIDX == 2) && (bitplane == 29))
//                        printf("ZC1 %d %d %x\n", TIDY, TIDX, sig);

//                if((TIDY == 3) && (TIDX == 2) && (bitplane == 29))
//	                printf("ZC2 %d %d %x\n", TIDY, TIDX, sig);

		unsigned int sig = nbh;
                sig |=  ((((coeff[Y - 1][X + 1] & SIGMA_NEW) & (nbh == 0)) << 2) | /*tr*/
                                (((coeff[Y][X + 1] & SIGMA_NEW) & (nbh == 0)) << 5) | /*r*/
                                (((coeff[Y + 1][X + 1] & SIGMA_NEW) & (nbh == 0)) << 8) | /*br*/
                                (((coeff[Y + 1][X] & SIGMA_NEW) & (nbh == 0)) << 7) | /*bc*/
                                (((coeff[Y + 1][X - 1] & SIGMA_NEW) & (nbh == 0)) << 6)); /*bl*/

		sig |= (((((coeff[Y - 1][X + 1] >> bitplane) & 1) & ((TIDY & 3) == 0x0) & (nbh == 0)) << 2) | /*tr*/
                       (((coeff[Y - 1][X] >> bitplane) & 1 & (nbh == 0)) << 1) | /*tc*/
                       (((coeff[Y - 1][X - 1] >> bitplane) & 1 & (nbh == 0)) << 0) | /*tl*/
                       (((coeff[Y][X - 1] >> bitplane) & 1 & (nbh == 0)) << 3) | /*l*/
                       ((((coeff[Y + 1][X - 1] >> bitplane) & 1) & ((TIDY & 3) != 0x3) & (nbh == 0)) << 6)); /*bl*/

		pairs |= getSPCX(sig, info->subband) << CX1_BITPOS; // set CX
		pairs |= ((coeff[Y][X] >> bitplane) & 1) << D1_BITPOS; // set D

//		if((TIDY == 3) && (TIDX == 2) && (bitplane == 29))
//                        printf("ZC %d %d %x %d %d\n", TIDY, TIDX, sig, (coeff[Y - 1][X + 1] & SIGMA_NEW), ((coeff[Y][X - 1] >> bitplane) & 1));

		save_cxd(cxds, pairs);
	//		cxds[TIDY][TIDX] = pairs;
	}
}

template <char Code_Block_Size_X>
__device__ void signCoding(unsigned int coeff[][Code_Block_Size_X + 2*BORDER], unsigned int cxds[][Code_Block_Size_X], unsigned int &pairs, const unsigned char bitplane) {
//	if σold = 0 AND bit value = 1 then
	if((!(coeff[Y][X] & SIGMA_OLD)) && ((coeff[Y][X] >> bitplane) & 1)) {
		// bitplane ?
		unsigned int sig_sign = (((coeff[Y - 1][X] & SIGMA_NEW) | ((coeff[Y - 1][X] & SIGMA_OLD) >> 1) | (((coeff[Y - 1][X] >> bitplane) & 1) && !(coeff[Y][X] & SIGMA_NEW))) << 7)/*V0*/ |
							((coeff[Y - 1][X] >> SIGN_BITPOS) << 6)/*V0*/ |
							(((coeff[Y][X - 1] & SIGMA_NEW) | ((coeff[Y][X - 1] & SIGMA_OLD) >> 1) | (((coeff[Y][X - 1] >> bitplane) & 1) && !(coeff[Y][X] & SIGMA_NEW))) << 5)/*H0*/ |
							((coeff[Y][X - 1] >> SIGN_BITPOS) << 4)/*H0*/ |
							((((coeff[Y][X + 1] & SIGMA_OLD) >> 1) | ((coeff[Y][X + 1] & SIGMA_NEW) && !(coeff[Y][X] & SIGMA_NEW))) << 3)/*H1*/ |
							((coeff[Y][X + 1] >> SIGN_BITPOS) << 2)/*H1*/ |
							((((coeff[Y + 1][X] & SIGMA_OLD) >> 1) | ((coeff[Y + 1][X] & SIGMA_NEW) && !(coeff[Y][X] & SIGMA_NEW))) << 1)/*V1*/ |
							((coeff[Y + 1][X] >> SIGN_BITPOS) << 0)/*V1*/;
		unsigned char cx_x = getSICX(sig_sign);
		unsigned char cx = cx_x & 0xF; // set CX
		unsigned char d = (coeff[Y][X] >> SIGN_BITPOS) ^ ((cx_x >> 4) & 1); // set D
		// shift by 6 (ZC), 18 (RLC on 1st bit), 0 (RLC on 2nd...4th bit)
		int shift = ((!(coeff[Y][X] & RLC)) * 6) + ((coeff[Y][X] & RLC) && (!(TIDY & 3))) * 18;
		pairs |= (cx << (D1_BITPOS + 1 - shift)); // save CX
		pairs |= (d << (D1_BITPOS - shift)); // save D
		pairs |= ((!(coeff[Y][X] & SIGMA_NEW)) << CUP_BITPOS) | ((coeff[Y][X] & SIGMA_NEW) << SPP_BITPOS); // set CUP or SPP, sigma_new differentiate
		coeff[Y][X] |= SIGMA_NEW;
//		if((TIDY == 15) && (TIDX == 12) && (bitplane == 29))
//                	printf("SC %d %d %x\n", TIDY, TIDX, (coeff[Y][X] & SIGMA_NEW));
//		if((TIDY == 4) && (TIDX == 1))
//			printf("SC %x %d %d\n", coeff[Y][X], TIDY, TIDX);
		save_cxd(cxds, pairs);
//		cxds[TIDY][TIDX] = pairs;
	}
}

template <char Code_Block_Size_X>
__global__ void bpc_encoder(CodeBlockAdditionalInfo *infos, unsigned int *g_cxds, const int maxOutLength) {
	// to access coeff use X and Y
	__shared__ unsigned int coeff[Code_Block_Size_X + 2*BORDER][Code_Block_Size_X + 2*BORDER];
	__shared__ unsigned int cxds[Code_Block_Size_X][Code_Block_Size_X];
	__shared__ unsigned int maxs[Code_Block_Size_X];
	__shared__ int blockVote;

	CodeBlockAdditionalInfo *info = &(infos[blockIdx.x]);
	unsigned char leastSignificantBP = 31 - info->magbits;
	if(info->magbits == 0) {
		if(TID == 0) info->significantBits = 0;
		return;
	}

	if((TIDX >= info->width) || (TIDY >= info->height)) return;

	// set borders to zero - not efficient way...
	if((TIDX < Code_Block_Size_X) && (TIDY == 0)) {
		coeff[0][TIDX + BORDER] = 0;
		coeff[info->height + BORDER][TIDX + BORDER] = 0;
	}

	if((TIDY < Code_Block_Size_X) && (TIDX == 0)) {
		coeff[TIDY + BORDER][0] = 0;
		coeff[TIDY + BORDER][info->width + BORDER] = 0;
	}

	if((TIDX == 0) && (TIDY == 0)) {
		coeff[0][0] = 0;
		coeff[info->height + BORDER][0] = 0;
		coeff[0][info->width + BORDER] = 0;
		coeff[info->width + BORDER][info->height + BORDER] = 0;
		blockVote = 0;
	}
	__syncthreads();

//	if((TIDX >= info->width) || (TIDY >= info->height)) return;

	int cache_value = info->coefficients[TIDY * info->nominalWidth + TIDX];
	coeff[Y][X] = cache_value/* < 0 ? (1 << 31) | (-cache_value) : cache_value*/;
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

	unsigned char significantBits = maxs[0] - leastSignificantBP + 1;
	if(TID == 0) info->significantBits = significantBits;

	// (x % (DIM /4)) * 4 + (y % 4)
	int bacx = (TIDX &((Code_Block_Size_X >> 2) - 1))*4 + (TIDY & 3);
	// floor(i /4) * 4 + floor(j /(DIM /4))
	int bacy = floorf(TIDY / 4)*4 + floorf(TIDX / (Code_Block_Size_X >> 2));

//	printf("%d\n", leastSignificantBP + significantBits - 1);
	// set cxds to zero, for RLC
        cxds[bacy][bacx] = 0;
        __syncthreads();

	cleanUpPassMSB<Code_Block_Size_X>(coeff, cxds, info, leastSignificantBP + significantBits - 1);
	__syncthreads();
	g_cxds[blockIdx.x * maxOutLength + bacy * info->height + bacx] = cxds[bacy][bacx];
	__syncthreads();

	int size = info->width * info->height;

	for(unsigned char i = 1; i < significantBits; ++i)
	{
		unsigned int pairs = 0;
		unsigned char bitplane = leastSignificantBP + significantBits - i - 1;
		blockVote = 0;
		// set cxds to zero, for RLC
        	//cxds[TIDY][TIDX] = 0;
		//__syncthreads();
		cxds[bacy][bacx] = 0;
	        __syncthreads();
		btiplanePreprocessing<Code_Block_Size_X>(coeff, blockVote, bitplane);
		__syncthreads();
		// MRP
		//if σold = 1
		magnitudeRefinementCoding<Code_Block_Size_X>(coeff, cxds, bitplane);
		__syncthreads();

		//rlcNbh := Σ(surrounding state variables)
		//RLC
		//if rlcNbh = 0 AND σold = 0 AND σnew = 0
		runLengthCoding<Code_Block_Size_X>(coeff, cxds, pairs, bitplane);
		__syncthreads();

		// ZC
		//if σold = 0 AND rlcNbh = 1 then
		//execute ZC operation
		zeroCoding<Code_Block_Size_X>(info, coeff, cxds, pairs, bitplane);
		__syncthreads();

		//SC
		//if σold = 0 AND bit value = 1 then
		//execute SC operation
		signCoding<Code_Block_Size_X>(coeff, cxds, pairs, bitplane);
		__syncthreads();

		//write to global memory
		g_cxds[blockIdx.x * maxOutLength + i * size + bacy * info->height + bacx] = cxds[bacy][bacx];
		__syncthreads();
	}
	__syncthreads();

//	if(bacy * info->height + bacx == 132)
//		printf("%x %d %d\n", cxds[bacy][bacx], TIDY, TIDX);
}

void launch_bpc_encode(dim3 gridDim, dim3 blockDim, CodeBlockAdditionalInfo *infos, unsigned int *g_cxds, const int maxOutLength)
{
	printf("dim %d %d\n", blockDim.x, blockDim.y);
	switch(blockDim.x) {
	case 4: bpc_encoder<4><<<gridDim, blockDim>>>(infos, g_cxds, maxOutLength); break;
	case 8: bpc_encoder<8><<<gridDim, blockDim>>>(infos, g_cxds, maxOutLength); break;
	case 16: bpc_encoder<16><<<gridDim, blockDim>>>(infos, g_cxds, maxOutLength); break;
	case 32: bpc_encoder<32><<<gridDim, blockDim>>>(infos, g_cxds, maxOutLength); break;
//	case 64: bpc_encoder<64><<<gridDim, blockDim>>>(infos, g_cxds); break;
	}

	cudaThreadSynchronize();
	cudaError_t cuerr;
	if (cuerr = cudaGetLastError()) {
		printf("bpc_encoder error: %s\n", cudaGetErrorString(cuerr));
		return;
	}
}
