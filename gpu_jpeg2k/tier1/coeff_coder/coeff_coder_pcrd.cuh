/* 
Copyright 2009-2013 Poznan Supercomputing and Networking Center

Authors:
Milosz Ciznicki miloszc@man.poznan.pl

GPU JPEG2K is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GPU JPEG2K is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with GPU JPEG2K. If not, see <http://www.gnu.org/licenses/>.
*/
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
