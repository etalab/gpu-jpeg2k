/*
 * coeff_coder_pcrd.cuh
 *
 *  Created on: Dec 8, 2011
 *      Author: miloszc
 */

#ifndef COEFF_CODER_PCRD_CUH_
#define COEFF_CODER_PCRD_CUH_

#include "gpu_coeff_coder2.cuh"

namespace GPU_JPEG2K
{
void launch_encode_pcrd(dim3 gridDim, dim3 blockDim, CoefficientState *coeffBuffors, byte *outbuf, int maxThreadBufforLength, CodeBlockAdditionalInfo *infos, int codeBlocks, int targetSize);
}

#endif /* COEFF_CODER_PCRD_CUH_ */
