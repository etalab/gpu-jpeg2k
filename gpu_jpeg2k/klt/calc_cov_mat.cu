/*
 * calculate_covariance_matrix.c
 *
 *  Created on: Dec 1, 2011
 *      Author: miloszc
 */
#include "calc_cov_mat.h"
#include "analysis.h"
extern "C" {
#include "calc_cov.h"
#include "../misc/memory_management.cuh"
#include "../print_info/print_info.h"
}

void calculate_cov_matrix(type_image *img, type_data** data, type_data* covMatrix) {
	calculate_covariance_matrix<<<img->num_components, img->num_components>>>(data, covMatrix, img->width*img->height, img->num_components);
}

void calculate_cov_matrix_new(type_image *img, type_data** data, type_data* covMatrix_d) {
	type_data *covMatrix_h = NULL;
	type_data *h_odata = NULL;
	type_data *d_o_data = NULL; /* device reduced output data */
	int n = img->num_components;
	int size = img->width * img->height;

	/* allocate data for output results on device memory */
	cuda_d_allocate_mem((void **) &d_o_data, sizeof(type_data) * MAX_BLOCKS);
	cuda_h_allocate_mem((void **)&covMatrix_h, n * n * sizeof(type_data));
	cuda_h_allocate_mem((void **)&h_odata, MAX_BLOCKS * sizeof(type_data));

	int i = 0, j = 0, counter = 0;
	for(i = 0; i < n; ++i) {
		for(j = 0; j < n; ++j) {
			if(j < i) {
//				cuda_d_memset(d_o_data, 0, sizeof(type_data) * MAX_BLOCKS);
				covMatrix_h[i * n + j] = calc_cov(data[i], data[j], d_o_data, h_odata, size);
				covMatrix_h[j * n + i] = covMatrix_h[i * n + j];
			}
		}
	}

	cuda_memcpy_htd(covMatrix_h, covMatrix_d, n * n * sizeof(type_data));
	cuda_h_free(covMatrix_h);
	cuda_d_free(d_o_data);
	cuda_h_free(h_odata);
}
