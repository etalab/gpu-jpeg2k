/*
 * calculate_covariance_matrix.c
 *
 *  Created on: Dec 1, 2011
 *      Author: miloszc
 */
#include "calc_cov_mat.h"
#include "analysis.h"
#include <cublas.h>
extern "C" {
#include "calc_cov.h"
#include "../misc/memory_management.cuh"
#include "../print_info/print_info.h"
}

void calculate_cov_matrix(type_image *img, type_data** data, type_data* covMatrix) {
	calculate_covariance_matrix<<<img->num_components, img->num_components>>>(data, covMatrix, img->width*img->height, img->num_components);
}

static void print_covMat(type_data* covMatrix_d, int n) {
	type_data *covMatrix_h = NULL;
	cuda_h_allocate_mem((void **)&covMatrix_h, n * n * sizeof(type_data));
	cuda_memcpy_dth(covMatrix_d, covMatrix_h, n * n * sizeof(type_data));

	int i = 0, j = 0, counter = 0;
	for(i = 0; i < 1; ++i) {
		for(j = 0; j < n; ++j) {
			printf("%f ", covMatrix_h[i * n + j]);
		}
		printf("\n");
	}
	cuda_h_free(covMatrix_h);
}

void calculate_cov_matrix_new(type_image *img, type_data** data, type_data* covMatrix_d) {
	type_data *covMatrix_h = NULL;
	type_data *h_odata = NULL;
	type_data *d_o_data = NULL; /* device reduced output data */
	int n = img->num_components;
	int size = img->width * img->height;

	cuda_d_allocate_mem((void **) &d_o_data, sizeof(type_data) * size * n);

	for(int i = 0; i < img->num_components; ++i) {
		cuda_memcpy_dtd((void *)data[i], (void *)(d_o_data + i * size), size * sizeof(type_data));
	}

	cublasSgemm('T', 'N', n, n, size, 1.0f, d_o_data, size, d_o_data, size, 0.0f, covMatrix_d, n);

	cublasSscal(n * n, 1.0f/(float)(size), covMatrix_d, 1);

/*	// allocate data for output results on device memory
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
	cuda_h_free(h_odata);*/
}
