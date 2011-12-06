#ifndef ANALYSIS_H_
#define ANALYSIS_H_

#include "../types/image_types.h"
#include "klt.h"

void __global__ calculate_covariance_matrix(type_data** data, type_data* covMatrix, int count, int dim);
void __global__ mean_adjust_data_g(type_data** data, type_data* means, int count);

#endif
