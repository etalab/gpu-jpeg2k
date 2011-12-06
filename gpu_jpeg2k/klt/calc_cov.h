/*
 * calc_cov.h
 *
 *  Created on: Dec 1, 2011
 *      Author: miloszc
 */

#ifndef CALC_COV_H_
#define CALC_COV_H_

#include "../types/image_types.h"

#define MAX_BLOCKS 65535

type_data calc_cov(type_data *d_i_data_i, type_data *d_i_data_j, type_data *d_o_data, type_data *h_odata, int size);

#endif /* CALC_COV_H_ */
