/*
 * test_calc_cov.cu
 *
 *  Created on: Dec 1, 2011
 *      Author: miloszc
 */
#include "../types/image_types.h"
extern "C" {
#include "calc_cov.h"
#include "../misc/memory_management.cuh"
}


void test_calc_cov() {
	type_data *h_i = NULL;
	type_data *h_j = NULL;
	int m = 4;
	cuda_h_allocate_mem((void **)&h_i, m * sizeof(type_data));
	cuda_h_allocate_mem((void **)&h_j, m * sizeof(type_data));

	int k = 0;
	for(k = 0; k < m; ++k) {
		h_i[k] = k + 1;
		h_j[k] = k + 1;
	}

	type_data *d_i = NULL;
	type_data *d_j = NULL;
	cuda_d_allocate_mem((void **)&d_i, m * sizeof(type_data));
	cuda_d_allocate_mem((void **)&d_j, m * sizeof(type_data));
	cuda_memcpy_htd(h_i, d_i, m * sizeof(type_data));
	cuda_memcpy_htd(h_j, d_j, m * sizeof(type_data));

	int res = calc_cov(d_i, d_j, m);
	printf("cov: %f == 30\n", res);
}
