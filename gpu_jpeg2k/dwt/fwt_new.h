/**
 * @file fwt.h
 *
 * @author Milosz Ciznicki
 */

#ifndef __FWT_NEW_H
#define __FWT_NEW_H

#include "dwt.h"

/*
  2D Forward DWT on tiles
*/

extern __global__ void fwt97_new(const float *idata, float *odata, const int2 img_size, const int2 step);

extern __global__ void fwt53_new(const float *idata, float *odata, const int2 img_size, const int2 step);

#endif

