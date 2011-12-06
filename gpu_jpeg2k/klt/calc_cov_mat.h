/*
 * calculate_covariance_matrix.h
 *
 *  Created on: Dec 1, 2011
 *      Author: miloszc
 */

#ifndef CALCULATE_COVARIANCE_MATRIX_H_
#define CALCULATE_COVARIANCE_MATRIX_H_

#include "../types/image_types.h"

void calculate_cov_matrix(type_image *img, type_data** data, type_data* covMatrix);
void calculate_cov_matrix_new(type_image *img, type_data** data, type_data* covMatrix_d);

#endif /* CALCULATE_COVARIANCE_MATRIX_H_ */
