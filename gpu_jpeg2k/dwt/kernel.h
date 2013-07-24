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
