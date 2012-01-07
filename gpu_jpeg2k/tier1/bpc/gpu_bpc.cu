/**
 * @file gpu_bpc.cu
 *
 * @author Milosz Ciznicki
 */

#include "gpu_bpc.h"

#define MAGBITS 0xEFFFF000

template <char Code_Block_Size_X>
__global__ void bpc_encoder(CodeBlockAdditionalInfo *infos) {
	__shared__ int coeff[Code_Block_Size_X][Code_Block_Size_X];
	__shared__ unsigned int cxds[Code_Block_Size_X][Code_Block_Size_X];
	__shared__ unsigned int maxs[Code_Block_Size_X];

	CodeBlockAdditionalInfo *info = &(infos[blockIdx.x]);

	coeff[threadIdx.y][threadIdx.x] = info->coefficients[threadIdx.y * info->nominalWidth + threadIdx.x];

	__syncthreads();

	// find most significant bitplane
	int tmp = 0;
	if((threadIdx.x < Code_Block_Size_X) && (threadIdx.y == 0)) {
		for(int i = 0; i < Code_Block_Size_X; ++i) {
			tmp = max(tmp, coeff[threadIdx.x][i] & MAGBITS);
		}
		maxs[threadIdx.x] = tmp;
	}
	__syncthreads();

	if(threadIdx.x == 0) {
		tmp = 0;
		for(int i = 0; i < Code_Block_Size_X; ++i) {
			tmp = max(tmp, maxs[i]);
		}
		maxs[0] = 31 - __clz(tmp);
		info->MSB = maxs[0];
	}
}

void launch_bpc_encode(dim3 gridDim, dim3 blockDim, CodeBlockAdditionalInfo *infos)
{
	bpc_encoder<16><<<gridDim, blockDim>>>(infos);
}
