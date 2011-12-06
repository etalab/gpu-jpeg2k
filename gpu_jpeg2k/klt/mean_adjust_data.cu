/*
 * mean_adjust_data.c
 *
 *  Created on: Nov 30, 2011
 *      Author: miloszc
 */
extern "C" {
#include "../misc/memory_management.cuh"
#include "../print_info/print_info.h"
#include "reduce.h"
}
#include "../misc/cuda_errors.h"
#include "analysis.h"
#include "shift.h"
#include "../types/image_types.h"
#include "mean_adjust_data.h"

void mean_adjust_data(type_image *img, type_data** data, type_data* means) {
	type_data* means_d;
	cuda_d_allocate_mem((void **) &means_d, sizeof(type_data)*img->num_components);

	mean_adjust_data_g<<< 1,img->num_components >>>(data, means_d, img->width*img->height);

	cuda_memcpy_dth((void *) means_d, (void *) means, sizeof(type_data)*img->num_components);
	cuda_d_free(means_d);
}

void mean_adjust_data_new(type_image *img, type_data** data, type_data* means) {
	int data_size = img->width * img->height;
	type_data *d_o_data = NULL; /* device reduced output data */
	type_data *h_odata = NULL;
	/* allocate data for output results on device memory */
	cuda_d_allocate_mem((void **) &d_o_data, sizeof(type_data) * MAX_BLOCKS);
	cuda_h_allocate_mem((void **)&h_odata, MAX_BLOCKS * sizeof(type_data));

	int i = 0;
	for(i = 0; i < img->num_components; ++i) {
		cuda_d_memset(d_o_data, 0, sizeof(type_data) * MAX_BLOCKS);
		means[i] = reduction(data[i], d_o_data, h_odata, data_size)/(type_data)data_size;
		shit(data[i], img->width, img->height, means[i]);
	}

	cuda_d_free(d_o_data);
	cuda_h_free(h_odata);
}
