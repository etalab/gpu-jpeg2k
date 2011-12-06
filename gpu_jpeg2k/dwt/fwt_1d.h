/*
 * @file fwt_1d.h
 *
 * @author Milosz Ciznicki 
 * @date 06-06-2011
 */

#ifndef FWT_1D_H_
#define FWT_1D_H_

#include "../types/image_types.h"

#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif

void fwt_1d(type_image *img, int lvl);

#endif /* FWT_1D_H_ */
