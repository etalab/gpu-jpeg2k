//#define DEBUG_MQ

#ifdef DEBUG_MQ
__device__ int l = 0;
__device__ int Cstates[1200000];
#endif

#include "gpu_coeff_coder2.cuh"
extern "C" {
	#include "../../misc/memory_management.cuh"
}
namespace GPU_JPEG2K
{
	#include "gpu_mq-coder.cuh"

__device__ void SetMaskedBits(unsigned int &reg, unsigned int mask, unsigned int bits)
{
	reg = (reg & ~mask) | (bits & mask);
}

__device__ void SetNthBit(unsigned int &reg, unsigned int n)
{
	SetMaskedBits(reg, 1 << n, 1 << n);
}

__device__ void ResetNthBit(unsigned int &reg, unsigned int n)
{
	SetMaskedBits(reg, 1 << n, 0);
}

typedef struct
{
	CoefficientState tl;
	CoefficientState t;
	CoefficientState tr;
	
	CoefficientState l;
	CoefficientState c;
	CoefficientState r;
	
	CoefficientState bl;
	CoefficientState b;
	CoefficientState br;

	short pos;
} CtxWindow;

typedef struct
{
	float dist;
	float slope;
	int L;
	int feasiblePoint;
} PcrdCodeblock;

typedef struct
{
	int feasibleNum;
	int Lfirst;
//	int nStates;
} PcrdCodeblockInfo;

__device__ void debug_print(float *val, int tid)
{
//	if(tid == 3)
//		printf("dist:%f\n", *val);
}


__device__ void down(CodeBlockAdditionalInfo &info, CtxWindow &window, CoefficientState *coeffs)
{
	window.tr = coeffs[window.pos + 1 - info.width];
	window.r = coeffs[window.pos + 1];
	window.br = coeffs[window.pos + 1 + info.width];
}

__device__ void up(CtxWindow &window, CoefficientState *coeffs)
{
	coeffs[window.pos - 1] = window.l;
}

__device__ void shift(CtxWindow &window)
{
	window.tl = window.t; window.t = window.tr; window.tr = 0; // top layer
	window.l = window.c; window.c = window.r; window.r = 0; // middle layer
	window.bl = window.b; window.b = window.br; window.br = 0; // bottom layer
	window.pos += 1;
}

typedef int CtxReg;

#define TRIMASK 0x249 //((1 << 0) | (1 << 3) | (1 << 6) | (1 << 9))

__device__ CtxReg buildCtxReg(CtxWindow &window, unsigned char bitoffset)
{
	CtxReg reg = 0;

	reg |= ((window.tl >> (bitoffset + 9)) & 1) << 0;
	reg |= ((window.t >> (bitoffset + 9)) & 1) << 1;
	reg |= ((window.tr >> (bitoffset + 9)) & 1) << 2;
	reg |= ((window.l >> (bitoffset)) & TRIMASK) << 3;
	reg |= ((window.c >> (bitoffset)) & TRIMASK) << 4;
	reg |= ((window.r >> (bitoffset)) & TRIMASK) << 5;
	reg |= ((window.bl >> (bitoffset)) & 1) << 15;
	reg |= ((window.b >> (bitoffset)) & 1) << 16;
	reg |= ((window.br >> (bitoffset)) & 1) << 17;
	
	return reg;
}

/*__device__ void updateWindow(CtxWindow &window, CtxReg reg, unsigned char bitoffset)
{
	window.tl = (window.tl & ~(1 << (bitoffset + 9)) | ((reg >> 0) & 1) << (bitoffset + 9);
	window.t = (window.t & ~(1 << (bitoffset + 9)) | ((reg >> 1) & 1) << (bitoffset + 9);
	window.tr = (window.tr & ~(1 << (bitoffset + 9)) | ((reg >> 2) & 1) << (bitoffset + 9);
	window.l = (window.l & ~(TRIMASK << (bitoffset)) | ((reg >> 3) & TRIMASK) << (bitoffset);
	window.c = (window.c & ~(TRIMASK << (bitoffset)) | ((reg >> 4) & TRIMASK) << (bitoffset);
	window.r = (window.r & ~(TRIMASK << (bitoffset)) | ((reg >> 5) & TRIMASK) << (bitoffset);
	window.bl = (window.bl & ~(1 << (bitoffset)) | ((reg >> 0) & 15) << (bitoffset);
	window.b = (window.b & ~(1 << (bitoffset)) | ((reg >> 0) & 16) << (bitoffset);
	window.br = (window.br & ~(1 << (bitoffset)) | ((reg >> 0) & 17) << (bitoffset);
}*/

	__constant__ unsigned char SPCXLUT[3][512] = {
		{
			0, 1, 3, 3, 1, 2, 3, 3, 5, 6, 7, 7, 6, 6, 7, 7, 0, 1, 3, 3,
			1, 2, 3, 3, 5, 6, 7, 7, 6, 6, 7, 7, 5, 6, 7, 7, 6, 6, 7, 7,
			8, 8, 8, 8, 8, 8, 8, 8, 5, 6, 7, 7, 6, 6, 7, 7, 8, 8, 8, 8,
			8, 8, 8, 8, 1, 2, 3, 3, 2, 2, 3, 3, 6, 6, 7, 7, 6, 6, 7, 7,
			1, 2, 3, 3, 2, 2, 3, 3, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7,
			6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 7, 7, 6, 6, 7, 7,
			8, 8, 8, 8, 8, 8, 8, 8, 3, 3, 4, 4, 3, 3, 4, 4, 7, 7, 7, 7,
			7, 7, 7, 7, 3, 3, 4, 4, 3, 3, 4, 4, 7, 7, 7, 7, 7, 7, 7, 7,
			7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7,
			7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 3, 3, 4, 4, 3, 3, 4, 4,
			7, 7, 7, 7, 7, 7, 7, 7, 3, 3, 4, 4, 3, 3, 4, 4, 7, 7, 7, 7,
			7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8,
			7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 1, 2, 3, 3,
			2, 2, 3, 3, 6, 6, 7, 7, 6, 6, 7, 7, 1, 2, 3, 3, 2, 2, 3, 3,
			6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 8, 8, 8, 8,
			8, 8, 8, 8, 6, 6, 7, 7, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8,
			2, 2, 3, 3, 2, 2, 3, 3, 6, 6, 7, 7, 6, 6, 7, 7, 2, 2, 3, 3,
			2, 2, 3, 3, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7,
			8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 7, 7, 6, 6, 7, 7, 8, 8, 8, 8,
			8, 8, 8, 8, 3, 3, 4, 4, 3, 3, 4, 4, 7, 7, 7, 7, 7, 7, 7, 7,
			3, 3, 4, 4, 3, 3, 4, 4, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
			7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7,
			8, 8, 8, 8, 8, 8, 8, 8, 3, 3, 4, 4, 3, 3, 4, 4, 7, 7, 7, 7,
			7, 7, 7, 7, 3, 3, 4, 4, 3, 3, 4, 4, 7, 7, 7, 7, 7, 7, 7, 7,
			7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7,
			7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8
		},
		{
			0, 1, 5, 6, 1, 2, 6, 6, 3, 3, 7, 7, 3, 3, 7, 7, 0, 1, 5, 6,
			1, 2, 6, 6, 3, 3, 7, 7, 3, 3, 7, 7, 3, 3, 7, 7, 3, 3, 7, 7,
			4, 4, 7, 7, 4, 4, 7, 7, 3, 3, 7, 7, 3, 3, 7, 7, 4, 4, 7, 7,
			4, 4, 7, 7, 1, 2, 6, 6, 2, 2, 6, 6, 3, 3, 7, 7, 3, 3, 7, 7,
			1, 2, 6, 6, 2, 2, 6, 6, 3, 3, 7, 7, 3, 3, 7, 7, 3, 3, 7, 7,
			3, 3, 7, 7, 4, 4, 7, 7, 4, 4, 7, 7, 3, 3, 7, 7, 3, 3, 7, 7,
			4, 4, 7, 7, 4, 4, 7, 7, 5, 6, 8, 8, 6, 6, 8, 8, 7, 7, 8, 8,
			7, 7, 8, 8, 5, 6, 8, 8, 6, 6, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8,
			7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8,
			7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 6, 6, 8, 8, 6, 6, 8, 8,
			7, 7, 8, 8, 7, 7, 8, 8, 6, 6, 8, 8, 6, 6, 8, 8, 7, 7, 8, 8,
			7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8,
			7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 1, 2, 6, 6,
			2, 2, 6, 6, 3, 3, 7, 7, 3, 3, 7, 7, 1, 2, 6, 6, 2, 2, 6, 6,
			3, 3, 7, 7, 3, 3, 7, 7, 3, 3, 7, 7, 3, 3, 7, 7, 4, 4, 7, 7,
			4, 4, 7, 7, 3, 3, 7, 7, 3, 3, 7, 7, 4, 4, 7, 7, 4, 4, 7, 7,
			2, 2, 6, 6, 2, 2, 6, 6, 3, 3, 7, 7, 3, 3, 7, 7, 2, 2, 6, 6,
			2, 2, 6, 6, 3, 3, 7, 7, 3, 3, 7, 7, 3, 3, 7, 7, 3, 3, 7, 7,
			4, 4, 7, 7, 4, 4, 7, 7, 3, 3, 7, 7, 3, 3, 7, 7, 4, 4, 7, 7,
			4, 4, 7, 7, 6, 6, 8, 8, 6, 6, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8,
			6, 6, 8, 8, 6, 6, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8,
			7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8,
			7, 7, 8, 8, 7, 7, 8, 8, 6, 6, 8, 8, 6, 6, 8, 8, 7, 7, 8, 8,
			7, 7, 8, 8, 6, 6, 8, 8, 6, 6, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8,
			7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8,
			7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8
		},
		{
			0, 3, 1, 4, 3, 6, 4, 7, 1, 4, 2, 5, 4, 7, 5, 7, 0, 3, 1, 4,
			3, 6, 4, 7, 1, 4, 2, 5, 4, 7, 5, 7, 1, 4, 2, 5, 4, 7, 5, 7,
			2, 5, 2, 5, 5, 7, 5, 7, 1, 4, 2, 5, 4, 7, 5, 7, 2, 5, 2, 5,
			5, 7, 5, 7, 3, 6, 4, 7, 6, 8, 7, 8, 4, 7, 5, 7, 7, 8, 7, 8,
			3, 6, 4, 7, 6, 8, 7, 8, 4, 7, 5, 7, 7, 8, 7, 8, 4, 7, 5, 7,
			7, 8, 7, 8, 5, 7, 5, 7, 7, 8, 7, 8, 4, 7, 5, 7, 7, 8, 7, 8,
			5, 7, 5, 7, 7, 8, 7, 8, 1, 4, 2, 5, 4, 7, 5, 7, 2, 5, 2, 5,
			5, 7, 5, 7, 1, 4, 2, 5, 4, 7, 5, 7, 2, 5, 2, 5, 5, 7, 5, 7,
			2, 5, 2, 5, 5, 7, 5, 7, 2, 5, 2, 5, 5, 7, 5, 7, 2, 5, 2, 5,
			5, 7, 5, 7, 2, 5, 2, 5, 5, 7, 5, 7, 4, 7, 5, 7, 7, 8, 7, 8,
			5, 7, 5, 7, 7, 8, 7, 8, 4, 7, 5, 7, 7, 8, 7, 8, 5, 7, 5, 7,
			7, 8, 7, 8, 5, 7, 5, 7, 7, 8, 7, 8, 5, 7, 5, 7, 7, 8, 7, 8,
			5, 7, 5, 7, 7, 8, 7, 8, 5, 7, 5, 7, 7, 8, 7, 8, 3, 6, 4, 7,
			6, 8, 7, 8, 4, 7, 5, 7, 7, 8, 7, 8, 3, 6, 4, 7, 6, 8, 7, 8,
			4, 7, 5, 7, 7, 8, 7, 8, 4, 7, 5, 7, 7, 8, 7, 8, 5, 7, 5, 7,
			7, 8, 7, 8, 4, 7, 5, 7, 7, 8, 7, 8, 5, 7, 5, 7, 7, 8, 7, 8,
			6, 8, 7, 8, 8, 8, 8, 8, 7, 8, 7, 8, 8, 8, 8, 8, 6, 8, 7, 8,
			8, 8, 8, 8, 7, 8, 7, 8, 8, 8, 8, 8, 7, 8, 7, 8, 8, 8, 8, 8,
			7, 8, 7, 8, 8, 8, 8, 8, 7, 8, 7, 8, 8, 8, 8, 8, 7, 8, 7, 8,
			8, 8, 8, 8, 4, 7, 5, 7, 7, 8, 7, 8, 5, 7, 5, 7, 7, 8, 7, 8,
			4, 7, 5, 7, 7, 8, 7, 8, 5, 7, 5, 7, 7, 8, 7, 8, 5, 7, 5, 7,
			7, 8, 7, 8, 5, 7, 5, 7, 7, 8, 7, 8, 5, 7, 5, 7, 7, 8, 7, 8,
			5, 7, 5, 7, 7, 8, 7, 8, 7, 8, 7, 8, 8, 8, 8, 8, 7, 8, 7, 8,
			8, 8, 8, 8, 7, 8, 7, 8, 8, 8, 8, 8, 7, 8, 7, 8, 8, 8, 8, 8,
			7, 8, 7, 8, 8, 8, 8, 8, 7, 8, 7, 8, 8, 8, 8, 8, 7, 8, 7, 8,
			8, 8, 8, 8, 7, 8, 7, 8, 8, 8, 8, 8
		}
	};

__device__ unsigned char getSPCX(CtxReg c, unsigned char i, unsigned char subband)
{
	return SPCXLUT[subband][(c >> (3 * i)) & 0x1FF];
}

	/* sign context in the following format
		index:
			first (MSB) bit V0 significance (1 significant, 0 insignificant)
			second bit V0 sign (0 positive, 1 negative)

			next 2 bits same for H0
			next 2 bits same for H1
			next 2 bits same for V1
			
		value:
			the response contains two pieces of information
			1. context label on the 4 least significant bits
			2. XORbit on the 5-th bit from the end (5-th least significant bit)
	*/

	__constant__ unsigned char signcxlut[256] = {
		 9,  9, 10, 26,  9,  9, 10, 26, 12, 12, 13, 11, 28, 28, 27, 29,  9,  9, 10, 26,
		 9,  9, 10, 26, 12, 12, 13, 11, 28, 28, 27, 29, 12, 12, 13, 11, 12, 12, 13, 11,
		12, 12, 13, 11,  9,  9, 10, 26, 28, 28, 27, 29, 28, 28, 27, 29,  9,  9, 10, 26,
		28, 28, 27, 29,  9,  9, 10, 26,  9,  9, 10, 26, 12, 12, 13, 11, 28, 28, 27, 29,
		 9,  9, 10, 26,  9,  9, 10, 26, 12, 12, 13, 11, 28, 28, 27, 29, 12, 12, 13, 11,
		12, 12, 13, 11, 12, 12, 13, 11,  9,  9, 10, 26, 28, 28, 27, 29, 28, 28, 27, 29,
		 9,  9, 10, 26, 28, 28, 27, 29, 10, 10, 10,  9, 10, 10, 10,  9, 13, 13, 13, 12,
		27, 27, 27, 28, 10, 10, 10,  9, 10, 10, 10,  9, 13, 13, 13, 12, 27, 27, 27, 28,
		13, 13, 13, 12, 13, 13, 13, 12, 13, 13, 13, 12, 10, 10, 10,  9, 27, 27, 27, 28,
		27, 27, 27, 28, 10, 10, 10,  9, 27, 27, 27, 28, 26, 26,  9, 26, 26, 26,  9, 26,
		11, 11, 12, 11, 29, 29, 28, 29, 26, 26,  9, 26, 26, 26,  9, 26, 11, 11, 12, 11,
		29, 29, 28, 29, 11, 11, 12, 11, 11, 11, 12, 11, 11, 11, 12, 11, 26, 26,  9, 26,
		29, 29, 28, 29, 29, 29, 28, 29, 26, 26,  9, 26, 29, 29, 28, 29
	};

__device__ unsigned char getSICX(CtxReg sig, CtxReg sign, unsigned char i)
{
	return signcxlut[
			((sig >> (i * 3)) & 0xAA) |
			(((sign >> (i * 3)) & 0xAA) >> 1)
		];
}

__device__ unsigned char getMRCX(CtxReg sig, CoefficientState local, unsigned char i)
{
	if((local >> (12 + 3 * i)) & 1)
		return 16;
	else if(((sig >> (3 * i)) & 0x1EF) == 0)
		return 14;
	else
		return 15;
}

template<class T> __device__ T min(T val1, T val2)
{
	if(val1 > val2)
	{
		return val2;
	} else
	{
		return val1;
	}
}

template<class T> __device__ T max(T val1, T val2)
{
	if(val1 < val2)
	{
		return val2;
	} else
	{
		return val1;
	}
}

__constant__ float distWeights[2][4][4] = {
{//Lossless
//		LH,      HL,      HH,     LLend
	{0.1000f, 0.1000f, 0.0500f, 1.0000f},  //level 0 = biggest subbands (unimportant)
	{0.2000f, 0.2000f, 0.1000f, 1.0000f},  //      1
	{0.4000f, 0.4000f, 0.2000f, 1.0000f},  //      2
	{0.8000f, 0.8000f, 0.4000f, 1.0000f}   //      3 = smallest, contains LL
}, {//Lossy
/*	{ 0.0010f, 0.0010f, 0.0005f, 1.0000f},
	{ 0.1000f, 0.1000f, 0.0250f, 1.0000f},
	{ 0.3000f, 0.3000f, 0.0800f, 1.0000f},
	{ 0.8000f, 0.8000f, 0.4000f, 1.0000f}*/
	{0.0100f, 0.0100f, 0.0050f, 1.0000f},
	{0.2000f, 0.2000f, 0.1000f, 1.0000f},
	{0.4000f, 0.4000f, 0.2000f, 1.0000f},
	{0.8000f, 0.8000f, 0.4000f, 1.0000f}
} };

__device__ float getDISW(CodeBlockAdditionalInfo *info)
{
	return distWeights[info->compType][min<byte>(info->dwtLevel, 3)][info->subband] * info->stepSize * info->stepSize / ((float)(info->width * info->height));
}

class RLEncodeFunctor {
public:
	__device__ char operator()(CtxWindow window, MQEncoder &enc)
	{
		char rest = 0;

		if((window.c & TRIMASK) == 0)
		{
			mqEncode(enc, 0, CX_RUN);
			rest = -2;
		}
		else
		{
			while(rest < 4 && ((window.c >> (3 * rest)) & 1) == 0)
				rest++;

			mqEncode(enc, 1, CX_RUN);
			mqEncode(enc, rest >> 1, CX_UNI);
			mqEncode(enc, rest & 1, CX_UNI);
		}

		return rest;
	}
};

class RLDecodeFunctor {
public:
	__device__ char operator()(CtxWindow &window, MQDecoder &dec)
	{
		char rest = 0;

		if(mqDecode(dec, CX_RUN) == 0)
		{
			rest = -2;
		}
		else
		{
			rest = mqDecode(dec, CX_UNI) & 1;
			rest <<= 1;
			rest |= mqDecode(dec, CX_UNI) & 1;

			window.c |= 1 << (3 * rest);
		}

		return rest;
	}
};

class SigEncodeFunctor {
public:
	__device__ void operator()(CtxWindow &window, CtxReg &sig, MQEncoder &enc, int stripId, int subband)
	{
		mqEncode(enc, (window.c >> (3 * stripId)) & 1, getSPCX(sig, stripId, subband));
	}
};

class SigDecodeFunctor {
public:
	__device__ void operator()(CtxWindow &window, CtxReg sig, MQDecoder &dec, int stripId, int subband)
	{
		window.c |= mqDecode(dec, getSPCX(sig, stripId, subband)) << (3 * stripId);
	}
};

class SignEncodeFunctor
{
public:
	__device__ void operator()(CtxWindow &window, CtxReg &sig, MQEncoder &enc, int stripId)
	{
		unsigned char cx = getSICX(sig, buildCtxReg(window, 13), stripId);

		mqEncode(enc, (short) (((window.c >> (13 + 3 * stripId)) & 1) ^ ((cx >> 4) & 1)), cx & 0xF);
	}
};

class SignDecodeFunctor
{
public:
	__device__ void operator()(CtxWindow &window, CtxReg sig, MQDecoder &dec, int stripId)
	{
		unsigned char cx = getSICX(sig, buildCtxReg(window, 13), stripId);

		window.c |= (mqDecode(dec, cx & 0xF) ^ ((cx >> 4) & 1) & 1) << (13 + 3 * stripId);
	}	
};

template <class RLCodingFunctor, class SigCodingFunctor, class SignCodingFunctor, typename MQCoderStateType>
class CleanUpPassFunctor
{
public:
	__device__ void operator()(const CodeBlockAdditionalInfo &info, CtxWindow &window, MQCoderStateType &mq, float *sum_dist, unsigned char bitplane)
	{
		char rest;

		CtxReg sig = buildCtxReg(window, 1); // significance context

		rest = -1;
		if((window.c & (TRIMASK << 14)) == 0 && sig == 0) // all contexts in stripe are equal to zero
		{
			rest = RLCodingFunctor()(window, mq);
			if(rest == -2)
				return;
		}

		for(int k = 0; k < 4; k++)
		{
			if(/*	((window.c >> ( 1 + 3 * k)) & 1) == 0 &&   // check if coefficient is non-significant (sigma)
				((window.c >> ( 2 + 3 * k)) & 1) == 0 &&   // check if coefficient hasn't been coded already (pi)
				((window.c >> (14 + 3 * k)) & 1) == 0)    // forbidden state indicating out of bounds (late sigma)*/
				((window.c >> (3 * k)) & 0x4006) == 0)
			{
				if(rest >= 0)
					rest--;
				else
					SigCodingFunctor()(window, sig, mq, k, info.subband);
			
				if((window.c >> (3 * k)) & 1) // check if magnitude is 1
				{
					*sum_dist -= (float)((1<<bitplane)*(1<<bitplane));
					debug_print(sum_dist, threadIdx.x);
//					if(blockIdx.x * blockDim.x + threadIdx.x == 0)
//					printf("clu:%f tid:%d\n", *sum_dist, blockIdx.x * blockDim.x + threadIdx.x);
					SetNthBit(window.c, 1 + 3 * k); // set k-th significant state
					sig = buildCtxReg(window, 1); // rebuild significance register

					SignCodingFunctor()(window, sig, mq, k);
				}
			}
		}
	}
};

template <class SigCodingFunctor, class SignCodingFunctor, typename MQCoderStateType>
class SigPropPassFunctor {
public:
__device__ void operator()(const CodeBlockAdditionalInfo &info, CtxWindow &window, MQCoderStateType &mq, float *sum_dist, unsigned char bitplane)
{
	CtxReg sig = buildCtxReg(window, 1); // build significance context register

	for(int i = 0; i < 4; i++)
	{
		// not significant with non-zero context
		if(/*	((window.c >> (1 + 3 * i)) & 1) == 0 &&
			((window.c >> (14 + 3 * i)) & 1) == 0 && // out of bounds
			getSPCX(sig, i, subband) > 0)*/
			(((window.c >> (3 * i)) & 0x4002) == 0) &&
			((sig >> (3 * i)) & 0x1EF) != 0)
		{
			SigCodingFunctor()(window, sig, mq, i, info.subband);

			// if magnitude bit is one
			if((window.c >> (3 * i)) & 1)
			{
				*sum_dist -= (float)((1<<bitplane)*(1<<bitplane));
				debug_print(sum_dist, threadIdx.x);
//				if(blockIdx.x * blockDim.x + threadIdx.x == 0)
//				printf("sig:%f tid:%d\n", *sum_dist, blockIdx.x * blockDim.x + threadIdx.x);
				SetNthBit(window.c, 1 + (3 * i));
				sig = buildCtxReg(window, 1); // rebuild

				SignCodingFunctor()(window, sig, mq, i);
			}

			// set pi (already coded)
			SetNthBit(window.c, 2 + (3 * i));
		}
		else
			// unset pi (already coded)
			ResetNthBit(window.c, 2 + (3 * i));
	}
}
};

class MagRefEncodeFunctor {
public:
	__device__ void operator()(MQEncoder &enc, CtxWindow &window, int stripId)
	{
		mqEncode(enc, (window.c >> (3 * stripId)) & 1, getMRCX(buildCtxReg(window, 1), window.c, stripId));
	}
};

class MagRefDecodeFunctor {
public:
	__device__ void operator()(MQDecoder &dec, CtxWindow &window, int stripId)
	{
		window.c |= (mqDecode(dec, getMRCX(buildCtxReg(window, 1), window.c, stripId)) << (3 * stripId));
	}
};

template <class MagRefCodingFunctor, typename MQCoderStateType>
class MagRefPassFunctor {
public:
__device__ void operator()(const CodeBlockAdditionalInfo &info, CtxWindow &window, MQCoderStateType &mq, float *sum_dist, unsigned char bitplane)
{
	for(int i = 0; i < 4; i++)
	{
		if(//csSignificant(st) && !csAlreadyCoded(st) && not out of bounds
			((window.c >> (3 * i)) & 0x4006) == 0x2)
		{
			*sum_dist -= (float)((1<<bitplane)*(1<<bitplane));
			debug_print(sum_dist, threadIdx.x);
//			if(blockIdx.x * blockDim.x + threadIdx.x == 0)
//			printf("mgr:%f tid:%d\n", *sum_dist, blockIdx.x * blockDim.x + threadIdx.x);
			MagRefCodingFunctor()(mq, window, i);
			SetNthBit(window.c, 3 * i + 12);
		}
	}
}
};

__device__ void initCoeffs(const CodeBlockAdditionalInfo &info, CoefficientState *coeffs)
{
	unsigned char signOffset = sizeof(int) * 8 - 1;

	for(int i = 0; i < info.width; i++)
		for(int j = 0; j < info.stripeNo; j++)
		{
			CoefficientState st = 0;
			int c;

			for(int k = 0; k < 4; k++)
				if(4 * j + k < info.height)
				{
					c = info.coefficients[(4 * j + k) * info.nominalWidth + i];
					//Cstates[l++] = (4 * j + k) * info.nominalWidth + i;
					st |= (((c >> signOffset) & 1) << (13 + 3 * k));
				}
				else
					st |= (1 << (14 + 3 * k));

			coeffs[j * info.width + i] = st;
		}
}

__device__ void initDecodingCoeffs(const CodeBlockAdditionalInfo &info, CoefficientState *coeffs)
{
	for(int i = 0; i < info.width; i++)
		for(int j = 0; j < info.stripeNo; j++)
		{
			CoefficientState st = 0;

			for(int k = 0; k < 4; k++)
				if(4 * j + k < info.height)
					info.coefficients[(4 * j + k) * info.nominalWidth + i] = 0;
				else
					st |= (1 << (14 + 3 * k));

			coeffs[j * info.width + i] = st;
		}
}

__device__ void uploadSigns(const CodeBlockAdditionalInfo &info, CoefficientState *coeffs)
{
	unsigned char signOffset = sizeof(int) * 8 - 1;

	for(int i = 0; i < info.width; i++)
		for(int j = 0; j < info.stripeNo; j++)
		{
			CoefficientState st = coeffs[j * info.width + i];

			for(int k = 0; k < 4; k++)
				if(((st >> (14 + 3 * k)) & 1) == 0)
					info.coefficients[(4 * j + k) * info.nominalWidth + i] |= (((st >> (13 + 3 * k)) & 1) << signOffset);

			coeffs[j * info.width + i] = st;
		}
}

__device__ void fillMags(const CodeBlockAdditionalInfo &info, CoefficientState *coeffs, int bitplane)
{
	for(int i = 0; i < info.width; i++)
		for(int j = 0; j < info.stripeNo; j++)
		{
			CoefficientState st = coeffs[j * info.width + i];

			// clear magnitudes and already coded flags
			st &= ~(TRIMASK | (TRIMASK << 2));
			//st |= ((st & (TRIMASK << 1)) << 11);

			for(int k = 0; k < 4; k++)
				if(((st >> (14 + 3 * k)) & 1) == 0)
					st |= ((info.coefficients[(4 * j + k) * info.nominalWidth + i] >> bitplane) & 1) << (3 * k);

			coeffs[j * info.width + i] = st;
		}
}

__device__ void uploadMags(const CodeBlockAdditionalInfo &info, CoefficientState *coeffs, int bitplane)
{
	for(int i = 0; i < info.width; i++)
		for(int j = 0; j < info.stripeNo; j++)
		{
			CoefficientState st = coeffs[j * info.width + i];

			for(int k = 0; k < 4; k++)
				if(((st >> (14 + 3 * k)) & 1) == 0)
					info.coefficients[(4 * j + k) * info.nominalWidth + i] |= (((st >> (3 * k)) & 1) << bitplane);

			// clear magnitudes and already coded flags
			st &= ~(TRIMASK | (TRIMASK << 2));

			coeffs[j * info.width + i] = st;
		}
}

__device__ void clearWindow(CtxWindow &w)
{
	w.bl = 0;
	w.b = 0;
	w.br = 0;

	w.l = 0;
	w.c = 0;
	w.r = 0;

	w.tl = 0;
	w.t = 0;
	w.tr = 0;
}

template <class PassFunctor, typename MQCoderStateType>
__device__ void BITPLANE_WINDOW_SCAN(CodeBlockAdditionalInfo &info, CoefficientState *coeffs, MQCoderStateType &enc, float *sum_dist, unsigned char bitplane) {
	CtxWindow window;

	window.pos = -1;

	for(int j = 0; j < info.stripeNo; j++)
	{
		clearWindow(window);
		down(info, window, coeffs);
		shift(window);
		down(info, window, coeffs);
	
		PassFunctor()(info, window, enc, sum_dist, bitplane);

		for(int k = 0; k < info.width - 2; k++)
		{
			shift(window);
			down(info, window, coeffs);
			PassFunctor()(info, window, enc, sum_dist, bitplane);
			up(window, coeffs);
		}

		shift(window);
		PassFunctor()(info, window, enc, sum_dist, bitplane);
		up(window, coeffs);
		shift(window);
		up(window, coeffs);

		window.pos--;
	}
}

class PCRD_EmptyFunctor
{
public:
	__device__ void operator()(MQEncoder state, MQEncoder *states, unsigned char &stateId, float sum_dist, PcrdCodeblock *pcrdCodeblock)
	{
	}
};

class PCRD_CollectMQStatesFunctor
{
public:
	__device__ void operator()(MQEncoder state, MQEncoder *states, unsigned char &stateId, float sum_dist, PcrdCodeblock *pcrdCodeblock)
	{
		states[stateId++] = state;
		pcrdCodeblock[stateId].dist = sum_dist;
	}
};

class CollectMQStatesFunctor
{
public:
	__device__ void operator()(MQEncoder state, MQEncoder *states, unsigned char &stateId, float sum_dist, PcrdCodeblock *pcrdCodeblock)
	{
		states[stateId++] = state;
	}
};

template <class PostPassFunctor, class PostCodingFunctor>
__device__ void encode(CoefficientState *coeffs, byte *out, CodeBlockAdditionalInfo &info, MQEncoder *states, PcrdCodeblock *pcrdCodeblock = NULL)
{
	unsigned char leastSignificantBP = 31 - info.magbits;

	info.significantBits = 0;
	int c;
	for(int i = 0; i < info.width; i++)
		for(int j = 0; j < info.height; j++)
		{
			c = info.coefficients[j * info.nominalWidth + i];
			int k;
			for(k = 30; k >= leastSignificantBP; k--)
				if((c >> k) & 1)
					break;

			if(k - leastSignificantBP + 1 > info.significantBits)
				info.significantBits = k - leastSignificantBP + 1;

		}

	MQEncoder mqenc;
	mqInitEnc(mqenc, out);
				
	unsigned char sid = 0;
	float sum_dist = 0.0f;

	if(pcrdCodeblock != NULL)
		pcrdCodeblock[sid].dist = 0;

	if(info.significantBits > 0)
	{
		mqResetEnc(mqenc);
		
		initCoeffs(info, coeffs);
		
		// first plane
		fillMags(info, coeffs, leastSignificantBP + info.significantBits - 1);
		
//		printf("bitplane:%d tid:%d\n", leastSignificantBP + info.significantBits - 1, blockIdx.x * blockDim.x + threadIdx.x);

		BITPLANE_WINDOW_SCAN
		<CleanUpPassFunctor<RLEncodeFunctor, SigEncodeFunctor, SignEncodeFunctor, MQEncoder>, MQEncoder >
			(info, coeffs, mqenc, &sum_dist, info.significantBits - 1);

		PostPassFunctor()(mqenc, states, sid, sum_dist, pcrdCodeblock);
		
		for(unsigned char i = 1; i < info.significantBits; i++)
		{
			fillMags(info, coeffs, leastSignificantBP + info.significantBits - i - 1);

//			printf("bitplane:%d tid:%d\n", leastSignificantBP + info.significantBits - i - 1, blockIdx.x * blockDim.x + threadIdx.x);
			
			BITPLANE_WINDOW_SCAN
			<SigPropPassFunctor<SigEncodeFunctor, SignEncodeFunctor, MQEncoder>, MQEncoder >
				(info, coeffs, mqenc, &sum_dist, info.significantBits - i - 1);

			PostPassFunctor()(mqenc, states, sid, sum_dist, pcrdCodeblock);

			BITPLANE_WINDOW_SCAN
			<MagRefPassFunctor<MagRefEncodeFunctor, MQEncoder>, MQEncoder >
				(info, coeffs, mqenc, &sum_dist, info.significantBits - i - 1);

			PostPassFunctor()(mqenc, states, sid, sum_dist, pcrdCodeblock);

			BITPLANE_WINDOW_SCAN
			<CleanUpPassFunctor<RLEncodeFunctor, SigEncodeFunctor, SignEncodeFunctor, MQEncoder>, MQEncoder >
				(info, coeffs, mqenc, &sum_dist, info.significantBits - i - 1);

			PostPassFunctor()(mqenc, states, sid, sum_dist, pcrdCodeblock);
		}

		PostCodingFunctor()(mqenc, states, sid, sum_dist, pcrdCodeblock);
		mqFlush(mqenc);
	}
}

__device__ void decode(CoefficientState *coeffs, CodeBlockAdditionalInfo &info, byte *in)
{
	MQDecoder mqdec;
	mqInitDec(mqdec, in, info.length);

	float sum_dist = 0.0f;

	if(info.significantBits > 0)
	{
		mqResetDec(mqdec);

		initDecodingCoeffs(info, coeffs);

		BITPLANE_WINDOW_SCAN
		<CleanUpPassFunctor<RLDecodeFunctor, SigDecodeFunctor, SignDecodeFunctor, MQDecoder>, MQDecoder>
			(info, coeffs, mqdec, &sum_dist, 0);

		uploadMags(info, coeffs, 30 - info.magbits + info.significantBits);

		for(unsigned char i = 1; i < info.significantBits; i++)
		{
			BITPLANE_WINDOW_SCAN
			<SigPropPassFunctor<SigDecodeFunctor, SignDecodeFunctor, MQDecoder>, MQDecoder>
				(info, coeffs, mqdec, &sum_dist, 0);

			BITPLANE_WINDOW_SCAN
			<MagRefPassFunctor<MagRefDecodeFunctor, MQDecoder>, MQDecoder>
				(info, coeffs, mqdec, &sum_dist, 0);

			BITPLANE_WINDOW_SCAN
			<CleanUpPassFunctor<RLDecodeFunctor, SigDecodeFunctor, SignDecodeFunctor, MQDecoder>, MQDecoder>
				(info, coeffs, mqdec, &sum_dist, 0);

			uploadMags(info, coeffs, 30 - info.magbits - i + info.significantBits);
		}

		uploadSigns(info, coeffs);
		//mqDecode(mqdec, CX_UNI);
	}
	else
	{
		for(int i = 0; i < info.height; i++)
			for(int j = 0; j < info.width; j++)
				info.coefficients[i * info.nominalWidth + j] = 0;
	}
}

__global__ void g_encode(CoefficientState *coeffBuffors, byte *outbuf, int maxThreadBufforLength, CodeBlockAdditionalInfo *infos, int codeBlocks, MQEncoder *mqstates)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if(threadId >= codeBlocks)
		return;

	CodeBlockAdditionalInfo info = infos[threadId];

	info.length = 0;
	encode<PCRD_EmptyFunctor, CollectMQStatesFunctor>(coeffBuffors + info.magconOffset, outbuf + threadId * maxThreadBufforLength, info, mqstates + threadId);

	infos[threadId].significantBits = info.significantBits;
	infos[threadId].length = info.length;
}

__global__ void g_encode_pcrd(CoefficientState *coeffBuffors, byte *outbuf, int maxThreadBufforLength, CodeBlockAdditionalInfo *infos, int codeBlocks, MQEncoder *mqstates, int maxStatesPerCodeblock, PcrdCodeblock *pcrdCodeblocks)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if(threadId >= codeBlocks)
		return;

	CodeBlockAdditionalInfo info = infos[threadId];

	info.length = 0;
	encode<PCRD_CollectMQStatesFunctor, PCRD_EmptyFunctor>(coeffBuffors + info.magconOffset, outbuf + threadId * maxThreadBufforLength, info, mqstates + threadId * maxStatesPerCodeblock, pcrdCodeblocks + threadId * maxStatesPerCodeblock);

	infos[threadId].significantBits = info.significantBits;
	infos[threadId].length = info.length;
}

__global__ void g_lengthCalculation(CodeBlockAdditionalInfo *infos, int codeBlocks, MQEncoder *mqstates)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if(threadId >= codeBlocks)
		return;

	if(infos[threadId].significantBits > 0) {
		infos[threadId].length = mqFullFlush(mqstates[threadId]);
		infos[threadId].codingPasses = infos[threadId].significantBits * 3 -2;
	}
	else {
		infos[threadId].length = 0;
		infos[threadId].codingPasses = 1;
	}
}

__global__ void g_lengthCalculation_pcrd(CodeBlockAdditionalInfo *infos, int codeBlocks, MQEncoder *mqstates, int maxStatesPerCodeBlock, PcrdCodeblock *pcrdCodeblocks, PcrdCodeblockInfo *pcrdCodeblockInfos)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if(threadId >= codeBlocks)
		return;

	mqstates += maxStatesPerCodeBlock * threadId;
	pcrdCodeblocks += maxStatesPerCodeBlock * threadId;
	pcrdCodeblockInfos += threadId;
	infos += threadId;

//	printf("%c[%d;%d;%dm", 27, 1, threadId * 2, 40);

	if(infos->significantBits > 0)
	{
		int nstates = (infos->significantBits - 1) * 3 + 1;
		
		int len;

//		pcrdCodeblockInfos->nStates = nstates;
		infos->codingPasses = nstates;

		pcrdCodeblocks[0].dist = 0;
		pcrdCodeblocks[0].L = 0;

		for(int i = 0; i < nstates; i++)
		{
			len = max<int>(mqFullFlush(mqstates[i]), 0);
//			mqstates[i].L = len;
			pcrdCodeblocks[i + 1].L = len;
			/*if(pcrdCodeblocks[i].L < 0)
			{
				printf("Error: Len < 0!\n");
			}*/
/*			if(threadId == 1)
			{
//				printf("L[%2d]:%6d %6f\n", i, pcrdCodeblocks[i].L, pcrdCodeblocks[i].dist);
			}*/
			pcrdCodeblocks[i + 1].dist *= ((len == 0) ? 0 : getDISW(infos));
//			pcrdCodeblocks[i + 1].dist *= getDISW(infos);
/*//			if(threadId == 1)
			{
//				printf("%f\n", getDISW(info));
//				printf("%d %f %d tid:%d\n", pcrdCodeblocks[i].L, pcrdCodeblocks[i].dist, pcrdCodeblockInfos->nStates, threadId);
				printf("L[%2d]:%6d %6f\n", i, pcrdCodeblocks[i].L, pcrdCodeblocks[i].dist);
			}*/
		}

		infos->length = len;
	}
	else
	{
//		printf("No significant bits!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
		mqstates[0].L = 0;
		pcrdCodeblocks[0].L = 0;
		infos->length = 0;
		infos->codingPasses = 1;
	}
}

__global__ void g_slopeCalculation(int codeBlocks, int maxStatesPerCodeBlock, PcrdCodeblock *pcrdCodeblocks, PcrdCodeblockInfo *pcrdCodeblockInfos, CodeBlockAdditionalInfo *infos, float *d_slope_max)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if(threadId >= codeBlocks)
//	if(threadId != 1)
		return;

	pcrdCodeblocks += maxStatesPerCodeBlock * threadId;
	pcrdCodeblockInfos += threadId;
	infos += threadId;

	int nFeasible = 0;
	int lastSlope = 0;
	float deltaD, deltaL;

	int nstates = infos->codingPasses;

	pcrdCodeblocks[nFeasible].slope = /*FLT_MAX*/1000000000;
	pcrdCodeblocks[nFeasible].feasiblePoint = 0;

	nFeasible = 1;
	pcrdCodeblockInfos->Lfirst = pcrdCodeblocks[nFeasible].L;

	for(int i = 1; i < nstates; i++)
	{
		deltaD = pcrdCodeblocks[lastSlope].dist - pcrdCodeblocks[i].dist;
		deltaL = (float)(pcrdCodeblocks[i].L - pcrdCodeblocks[lastSlope].L);

//		if(threadId == 12)
//		printf("%f %f deltaD:%f slope*deltaL:%f L[%d]:%d last[%d]:%d deltaL:%f\n", pcrdCodeblocks[lastSlope].dist , pcrdCodeblocks[i].dist, deltaD, pcrdCodeblocks[lastSlope].slope * deltaL, i, pcrdCodeblocks[i].L, lastSlope, pcrdCodeblocks[lastSlope].L, deltaL);

		if(deltaD > 0.0f)
		{
			while(deltaD >= pcrdCodeblocks[lastSlope].slope * deltaL)
			{
				/*if(nFeasible == 0)
				{
					printf("nFeasible:%d tid:%d\n", nFeasible, threadId);
					return;
				}*/
				nFeasible--;
				lastSlope = pcrdCodeblocks[nFeasible - 1].feasiblePoint;
				deltaD = pcrdCodeblocks[lastSlope].dist - pcrdCodeblocks[i].dist;
				deltaL = (float)(pcrdCodeblocks[i].L - pcrdCodeblocks[lastSlope].L);
			}
			lastSlope = i;
			pcrdCodeblocks[nFeasible++].feasiblePoint = lastSlope;
			pcrdCodeblocks[lastSlope].slope = deltaD/deltaL;
		}
	}

	pcrdCodeblockInfos->feasibleNum = nFeasible;

	int feasiblePoint;

	for(int i = 0; i < nFeasible; i++)
	{
		feasiblePoint = pcrdCodeblocks[i].feasiblePoint;
		pcrdCodeblocks[i].L = pcrdCodeblocks[feasiblePoint].L;
		pcrdCodeblocks[i].slope = pcrdCodeblocks[feasiblePoint].slope;

//		printf("%f\n", *d_slope_max);
		if((pcrdCodeblocks[i].slope > *d_slope_max) && (i != 0))
		{
			*d_slope_max = pcrdCodeblocks[i].slope;
		}
//		printf("cblk:%2d pcrdCodeblockInfos->Lfirst:%3d feasiblePoints:%6d truncLengths[i]:%6d slopes:%6f\n", threadId, pcrdCodeblockInfos->Lfirst, pcrdCodeblockInfos->feasibleNum, i, pcrdCodeblocks[i].L, pcrdCodeblocks[i].slope);
	}
}

__global__ void g_decode(CoefficientState *coeffBuffors, byte *inbuf, int maxThreadBufforLength, CodeBlockAdditionalInfo *infos, int codeBlocks)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if(threadId >= codeBlocks)
		return;

	CodeBlockAdditionalInfo info = infos[threadId];

	decode(coeffBuffors + info.magconOffset, info, inbuf + threadId * maxThreadBufforLength);
}

#include <stdio.h>

void launch_encode(dim3 gridDim, dim3 blockDim, CoefficientState *coeffBuffors, byte *outbuf, int maxThreadBufforLength, CodeBlockAdditionalInfo *infos, int codeBlocks)
{
	MQEncoder *mqstates;
	cuda_d_allocate_mem((void **) &mqstates, sizeof(MQEncoder) * codeBlocks);

//	printf("grid %d %d %d\nblock %d %d %d\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);

	g_encode<<<gridDim, blockDim>>>(coeffBuffors, outbuf, maxThreadBufforLength, infos, codeBlocks, mqstates);

	g_lengthCalculation<<<(int) ceil(codeBlocks / 512.0f), 512>>>(infos, codeBlocks, mqstates);

	cuda_d_free(mqstates);
}

#define CBLKS_PER_THREAD 16
#define THREADS_PER_BLOCK 4

__global__ void g_truncateSize(int codeBlocks, int maxStatesPerCodeBlock, CodeBlockAdditionalInfo *infos, PcrdCodeblock *pcrdCodeblocks, PcrdCodeblockInfo *pcrdCodeblockInfos, float lambda, int *sizes)
{
	int threadId = threadIdx.x + blockIdx.x * blockDim.x;
	int cbStart = threadId * CBLKS_PER_THREAD;
	int cbStop = (threadId + 1) * CBLKS_PER_THREAD > codeBlocks ? codeBlocks : (threadId + 1) * CBLKS_PER_THREAD;
	int i, size = 0;

//	printf("cbStart:%d cbStop:%d\n", cbStart, cbStop);

	for(i = cbStart; i < cbStop; ++i)
	{
		CodeBlockAdditionalInfo * info = &(infos[i]);
		PcrdCodeblock *pcrdCodeblock = &(pcrdCodeblocks[maxStatesPerCodeBlock * i]);
		PcrdCodeblockInfo *pcrdCodeblockInfo = &(pcrdCodeblockInfos[i]);
		/*infos += i;
		pcrdCodeblocks += maxStatesPerCodeBlock * i;
		pcrdCodeblockInfos += i;*/

		int j = 0;

		while((j + 1 < pcrdCodeblockInfo->feasibleNum) && (pcrdCodeblock[j + 1].slope > lambda)) j++;

		if(j > 0)
		{
			info->codingPasses = pcrdCodeblock[j].feasiblePoint;
			info->length = pcrdCodeblock[j].L;
			size += pcrdCodeblock[j].L;
//			printf("j > 0 cblk:%d size:%d\n", i, size);
		} else
		{
			info->codingPasses = 1;
			info->length = pcrdCodeblockInfo->Lfirst;
			size += pcrdCodeblockInfo->Lfirst;
//			printf("j <= 0 cblk:%d pcrdCodeblockInfos->Lfirst:%d size:%d\n", i, pcrdCodeblockInfo->Lfirst, size);
		}
	}

	if(cbStart < codeBlocks)
		sizes[threadId] = size;
}

int truncateSize(int codeBlocks, int maxStatesPerCodeBlock, CodeBlockAdditionalInfo *infos, PcrdCodeblock *pcrdCodeblocks, PcrdCodeblockInfo *pcrdCodeblockInfos, float lambda)
{
	int threads = (codeBlocks + CBLKS_PER_THREAD - 1) / CBLKS_PER_THREAD;
	int blocks = (threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	int size = 0;

	int *sizes_h, *sizes_d;

	cuda_d_allocate_mem((void **)&sizes_d, sizeof(int) * threads);
	cuda_d_memset(sizes_d, 0, sizeof(int) * threads);
	cuda_h_allocate_mem((void **)&sizes_h, sizeof(int) * threads);

//	printf("threads:%d blocks:%d\n", threads, blocks);

	g_truncateSize<<<blocks, THREADS_PER_BLOCK>>>(codeBlocks, maxStatesPerCodeBlock, infos, pcrdCodeblocks, pcrdCodeblockInfos, lambda, sizes_d);

	cudaThreadSynchronize();

	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", "g_truncateSize", cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}

	cuda_memcpy_dth(sizes_d, sizes_h, sizeof(int) * threads);

	int i;

	for(i = 0; i < threads; ++i)
	{
//		printf("sizes_h[%d]:%d\n", i, sizes_h[i]);
		size += sizes_h[i];
	}

	cuda_d_free(sizes_d);
	cuda_h_free(sizes_h);

	return size;
}

#define ALLOWED_DIFF 400.0f

void launch_pcrd(int maxStatesPerCodeBlock, int targetSize, float slopeMax, int codeBlocks, CodeBlockAdditionalInfo *infos, PcrdCodeblock *pcrdCodeblocks, PcrdCodeblockInfo *pcrdCodeblockInfos)
{
	float lambdaMin = -1.0f;
	float lambdaMax = slopeMax * 2.0f;
	float lambdaMid;
	int overHead, minSize, maxSize;

	//TODO
	overHead = 181 + /*numTiles *//*3 * 14 + */codeBlocks * 2;
//	printf("over head:%d\n", overHead);
//	overHead = 0;

	minSize = overHead + truncateSize(codeBlocks, maxStatesPerCodeBlock, infos, pcrdCodeblocks, pcrdCodeblockInfos, lambdaMax);

//	printf("minSize:%d\n", minSize);

	if(targetSize <= minSize)
	{
		printf("Target size to small %d using %d\n", targetSize, minSize);
		return;
	}

	maxSize = overHead + truncateSize(codeBlocks, maxStatesPerCodeBlock, infos, pcrdCodeblocks, pcrdCodeblockInfos, lambdaMin);

//	printf("maxSize:%d\n", maxSize);

	if(targetSize >= maxSize)
	{
		printf("Target size to large %d using %d\n", targetSize, maxSize);
		return;
	}

	float allowedDiff = ALLOWED_DIFF > (0.02f * (float)targetSize) ? ALLOWED_DIFF : (0.02f * (float)targetSize);
	int size, iterations = 0, countRefine = 0;

	do {
		lambdaMid = 0.5f * (lambdaMin + lambdaMax);

		size = overHead + truncateSize(codeBlocks, maxStatesPerCodeBlock, infos, pcrdCodeblocks, pcrdCodeblockInfos, lambdaMid);

//		printf("size:%d\n", size);

		if(size < targetSize)
			lambdaMax = lambdaMid;
		else
			lambdaMin = lambdaMid;

		if(countRefine == 0)
		{
			if(abs(targetSize - size) < allowedDiff)
			{
				countRefine = 1;
//				printf("allowedDiff:%d\n", abs(targetSize - size));
			}
		} else
		{
			countRefine++;
		}

		iterations++;
	} while(countRefine < 20 && iterations < 50);
}

void launch_encode_pcrd(dim3 gridDim, dim3 blockDim, CoefficientState *coeffBuffors, byte *outbuf, int maxThreadBufforLength, CodeBlockAdditionalInfo *infos, int codeBlocks, int targetSize)
{
	const int maxMQStatesPerCodeBlock = (MAX_MAG_BITS - 1) * 3 + 1;

	MQEncoder *mqstates;
	cuda_d_allocate_mem((void **) &mqstates, sizeof(MQEncoder) * codeBlocks * maxMQStatesPerCodeBlock);

	PcrdCodeblock *pcrdCodeblocks;
	cuda_d_allocate_mem((void **) &pcrdCodeblocks, sizeof(PcrdCodeblock) * codeBlocks * maxMQStatesPerCodeBlock);
	cuda_d_memset((void *)pcrdCodeblocks, 0, sizeof(PcrdCodeblock) * codeBlocks * maxMQStatesPerCodeBlock);

	g_encode_pcrd<<<gridDim, blockDim>>>(coeffBuffors, outbuf, maxThreadBufforLength, infos, codeBlocks, mqstates, maxMQStatesPerCodeBlock, pcrdCodeblocks);

	PcrdCodeblockInfo *pcrdCodeblockInfos;
	cuda_d_allocate_mem((void **) &pcrdCodeblockInfos, sizeof(PcrdCodeblockInfo) * codeBlocks);

	g_lengthCalculation_pcrd<<<(int) ceil(codeBlocks / 512.0f), 512>>>(infos, codeBlocks, mqstates, maxMQStatesPerCodeBlock, pcrdCodeblocks, pcrdCodeblockInfos);

	cuda_d_free(mqstates);

	float *dSlopeMax;
	cuda_d_allocate_mem((void**)&dSlopeMax, sizeof(float));
	cuda_d_memset((void *)dSlopeMax, 0, sizeof(float));

	g_slopeCalculation<<<(int) ceil(codeBlocks / 512.0f), 512>>>(codeBlocks, maxMQStatesPerCodeBlock, pcrdCodeblocks, pcrdCodeblockInfos, infos, dSlopeMax);

	//TODO debug
	PcrdCodeblockInfo *pcrdCodeblockInfos_h;
	pcrdCodeblockInfos_h = (PcrdCodeblockInfo *) malloc(sizeof(PcrdCodeblockInfo) * codeBlocks);
	cuda_memcpy_dth(pcrdCodeblockInfos, pcrdCodeblockInfos_h, sizeof(PcrdCodeblockInfo) * codeBlocks);

	/*for(int i = 0; i < codeBlocks; ++i)
	{
		printf("cblk:%d feasibleNum:%d Lfirst:%d\n", i, pcrdCodeblockInfos_h[i].feasibleNum, pcrdCodeblockInfos_h[i].Lfirst);
	}*/

	float slopeMax;
	cuda_memcpy_dth(dSlopeMax, &slopeMax, sizeof(float));

	//TODO
//	int targetSize = 380;

	launch_pcrd(maxMQStatesPerCodeBlock, targetSize, slopeMax, codeBlocks, infos, pcrdCodeblocks, pcrdCodeblockInfos);

	cuda_d_free(pcrdCodeblocks);
	cuda_d_free(pcrdCodeblockInfos);

//	launch_pcrd(infos, codeBlocks, mqstates, maxMQStatesPerCodeBlock, pcrd);
}

void launch_decode(dim3 gridDim, dim3 blockDim, CoefficientState *coeffBuffors, byte *inbuf, int maxThreadBufforLength, CodeBlockAdditionalInfo *infos, int codeBlocks)
{
	g_decode<<<gridDim, blockDim>>>(coeffBuffors, inbuf, maxThreadBufforLength, infos, codeBlocks);
}

}
