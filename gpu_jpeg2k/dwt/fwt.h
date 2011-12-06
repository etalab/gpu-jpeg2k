/**
 * @file fwt.h
 *
 * @author Milosz Ciznicki
 */

#ifndef __FWT_H
#define __FWT_H

#include "dwt.h"

/*
  2D Forward DWT on tiles
*/

extern __global__ void fwt97(const float *idata, float *odata, const int2 img_size, const int2 step);

extern __global__ void fwt53(const float *idata, float *odata, const int2 img_size, const int2 step);

#endif

