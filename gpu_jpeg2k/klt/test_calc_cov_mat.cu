/*
 * test_calc_cov_mat.cu
 *
 *  Created on: Dec 1, 2011
 *      Author: miloszc
 */

extern "C" {
#include "calc_cov.h"
#include "../misc/memory_management.cuh"
#include "../print_info/print_info.h"
}

void test_calc_cov_mat(type_image *img, type_data *covMatrix_d) {
	type_data *covMat_h = NULL;
	cuda_h_allocate_mem((void **) &covMat_h, sizeof(type_data) * img->num_components * img->num_components);
	cuda_memcpy_dth(covMatrix_d, covMat_h, sizeof(type_data) * img->num_components * img->num_components);

	int k, l;
	double cov_sum = 0;
	for (k = 0; k < img->num_components; ++k) {
		for (l = 0; l < img->num_components; ++l) {
			if (l < k) {
				cov_sum += covMat_h[k * img->num_components + l];
				//				fprintf(stdout, "%.1f ", covMat_h[k * img->num_components + l]);
			}
		}
		//		printf("\n");
	}
	printf("cov_sum %f\n", cov_sum);
}
