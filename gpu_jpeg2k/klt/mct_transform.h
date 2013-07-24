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
 * mct_transform.h
 *
 *  Created on: Nov 30, 2011
 *      Author: miloszc
 */

#ifndef MCT_TRANSFORM_H_
#define MCT_TRANSFORM_H_

#include "../types/image_types.h"
#define THREADS 256

void mct_transform(type_image *img, type_data* transform_d, type_data **data_pd, int odciecie);
void mct_transform_new(type_image *img, type_data* transform_d, type_data **data_pd, int odciecie);

#endif /* MCT_TRANSFORM_H_ */
