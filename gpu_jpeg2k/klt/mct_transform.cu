/* 
Copyright 2009-2013 Poznan Supercomputing and Networking Center

Authors:
Milosz Ciznicki miloszc@man.poznan.pl

GPU JPEG2K is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GPU JPEG2K is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with GPU JPEG2K. If not, see <http://www.gnu.org/licenses/>.
*/
/*
 * mct_transform.c
 *
 *  Created on: Nov 30, 2011
 *      Author: miloszc
 */

extern "C" {
#include "../misc/memory_management.cuh"
#include "../print_info/print_info.h"
}
#include "mct_transform.h"
#include "adjust.h"
#include "../misc/cuda_errors.h"

void mct_transform(type_image *img, type_data* transform_d, type_data **data_pd, int odciecie) {
	int i = 0;
	type_data* data_d;
	type_data* inter_d;
	cuda_d_allocate_mem((void **) &data_d, img->num_components * sizeof(type_data));
	cuda_d_allocate_mem((void **) &inter_d, (img->num_components - odciecie) * sizeof(type_data));

	for(i=0; i<img->width*img->height; ++i) {
	cudaThreadSynchronize();
		readSampleSimple<<<1,img->num_components>>>(data_pd, i, data_d);
		checkCUDAError("\tafter data read");
	//	println_var(INFO, "\tsample %d read",i);
	cudaThreadSynchronize();
		adjust_pca_data_mm(transform_d, FORWARD, data_d, inter_d, img->num_components, img->num_components);
		checkCUDAError("\tafter data transform");
	//	println_var(INFO, "\tsample %d transform",i);
	cudaThreadSynchronize();
		writeSampleSimple<<<1,img->num_components-odciecie>>>(data_pd, i, inter_d);
		checkCUDAError("\tafter data write");
	//	println_var(INFO, "\tsample %d write",i);
	}
}

void mct_transform_new(type_image *img, type_data* transform_d, type_data **data_pd, int odciecie) {
	int i = 0;
	int num_vecs = img->width * img->height;
	int len_vec = img->num_components;
	type_data* data_d;
	type_data* inter_d;
	cuda_d_allocate_mem((void **) &data_d, num_vecs * len_vec * sizeof(type_data));
	cuda_d_allocate_mem((void **) &inter_d, num_vecs * len_vec * sizeof(type_data));

	int blocks = (num_vecs + (THREADS - 1))/THREADS;

	readSamples<<<blocks, THREADS>>>(data_pd, num_vecs, len_vec, data_d);

	for(i = 0; i < num_vecs; ++i) {
		type_data *i_vec = data_d + i * len_vec;
		type_data *o_vec = inter_d + i * len_vec;
		adjust_pca_data_mv(transform_d, FORWARD, i_vec, o_vec, len_vec, len_vec);
	}

	writeSamples<<<blocks, THREADS>>>(data_pd, num_vecs, len_vec, inter_d);

	cuda_d_free(data_d);
	cuda_d_free(inter_d);
}
