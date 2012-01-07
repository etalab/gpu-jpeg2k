/**
 * @file gpu_bpc.h
 *
 * @author Milosz Ciznicki
 */

#ifndef GPU_BPC_H_
#define GPU_BPC_H_

#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif

typedef struct
{
	// temporary cdx length
	int length;
	unsigned char significantBits;
	unsigned char codingPasses;
	unsigned char width;
	unsigned char nominalWidth;
	unsigned char height;
	unsigned char stripeNo;
	unsigned char magbits;
	unsigned char subband;
	unsigned char compType;
	unsigned char dwtLevel;
	float stepSize;

	int magconOffset;

	int MSB;

	int* coefficients;
} CodeBlockAdditionalInfo;

void launch_bpc_encode(dim3 gridDim, dim3 blockDim, CodeBlockAdditionalInfo *infos);

#endif /* GPU_BPC_H_ */
