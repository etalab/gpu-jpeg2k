/*
 * gpu_convert.h
 *
 *  Created on: Feb 20, 2012
 *      Author: miloszc
 */

#ifndef GPU_CONVERT_H_
#define GPU_CONVERT_H_

void convert(dim3 gridDim, dim3 blockDim, CodeBlockAdditionalInfo *infos, unsigned int *g_icxds, unsigned char *g_ocxds, const int maxOutLength);


#endif /* GPU_CONVERT_H_ */
