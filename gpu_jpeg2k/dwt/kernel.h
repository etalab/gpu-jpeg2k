/**
 * @file kernel.h
 *
 * @author Milosz Ciznicki
 */

#ifndef __KERNEL_H
#define __KERNEL_H

#define DWT53	0
#define DWT97	1

#include "../types/image_types.h"

/*
	Perform the forward wavelet transform on a 2D matrix
*/
extern type_data *fwt_2d(short filter, type_tile_comp *tile_comp);

/*
	Perform the inverse wavelet transform on a 2D matrix
*/
type_data *iwt_2d(short filter, type_tile_comp *tile_comp);
//void iwt_2d(short filter, float *d_idata, float *d_odata, int2 img_size, const int2 step, const int nlevels);

#endif
