/*
 * reduce.h
 *
 *  Created on: Nov 30, 2011
 *      Author: miloszc
 */

#ifndef REDUCE_H_
#define REDUCE_H_

#include "../types/image_types.h"

#define MAX_BLOCKS 65535

type_data reduction(type_data *d_i_data, type_data *d_o_data, type_data *h_odata, int size);

#endif /* REDUCE_H_ */
