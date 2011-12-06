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
