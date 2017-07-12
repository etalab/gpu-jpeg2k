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
extern "C" {
#include "klt.h"
#include "../misc/memory_management.cuh"
#include "../print_info/print_info.h"
}
#include "gs.h"
#include "analysis.h"
#include "adjust.h"
#include <time.h>
#include "../my_common/my_common.h"
#include "../misc/cuda_errors.h"
#include "mct_transform.h"
#include "mean_adjust_data.h"
#include "calc_cov_mat.h"

#define MCT_MEAN_ADJUST_DATA
#define MCT_CALC_COV_MATRIX
#define MCT_CALC_GS
#define MCT_DROP_COMPONENTS
#define MCT_TRANSFORM_DATA
//#define MCT_FREEUP_COMPONENTS // now i know it is part of larger block and it is advised not to free this memory at this point xP
#define PART_TIME

void encode_klt(type_parameters *param, type_image *img) {
//	checkCUDAError("before MCT");
	long int start_klt;
	start_klt = start_measure();
	clock_t start = clock();

#ifdef PART_TIME
	long int start_prepare_data;
	start_prepare_data = start_measure();
#endif

	/* preparing pointers to components data */

	type_data** data_pd;
	type_data** data_p;
	
	cuda_d_allocate_mem((void **) &data_pd, sizeof(type_data*)*img->num_components);
	cuda_h_allocate_mem((void **) &data_p, img->num_components* sizeof(type_data*));
	for(int g=0; g<img->num_components; ++g) {
		data_p[g] = img->tile[0].tile_comp[g].img_data_d;
	}
	cuda_memcpy_htd((void *) data_p, (void*) data_pd, sizeof(type_data*)*img->num_components);

#ifdef PART_TIME
	cudaThreadSynchronize();
	printf("prepare data: %ld\n", stop_measure(start_prepare_data));
	long int start_mean_adj_comp;
	start_mean_adj_comp = start_measure();
#endif

	/* mean adjust components data */

//	checkCUDAError("before adjust");
	type_data* means;
	cuda_h_allocate_mem((void **) &means, sizeof(type_data)*img->num_components);
#ifdef MCT_MEAN_ADJUST_DATA
//	mean_adjust_data(img, data_pd, means);
	mean_adjust_data_new(img, data_p, means);
#endif
//	checkCUDAError("after mean adjust");
//	println_var(INFO, "after mans adjust");
	
#ifdef PART_TIME
	cudaThreadSynchronize();
	printf("mean adjust comp: %ld\n", stop_measure(start_mean_adj_comp));
	long int start_calc_cov_mat;
	start_calc_cov_mat = start_measure();
#endif

	/* calculate covariance matrix */
	
	type_data* covMatrix_d;
	cuda_d_allocate_mem((void **) &covMatrix_d, sizeof(type_data)*img->num_components*img->num_components);

#ifdef MCT_MEAN_ADJUST_DATA
//	calculate_cov_matrix(img, data_pd, covMatrix_d);
	calculate_cov_matrix_new(img, data_p, covMatrix_d);
#endif

	type_data* eValues;
	cuda_h_allocate_mem((void**) &eValues, img->num_components * sizeof(type_data));
	type_data* output; 
	cuda_h_allocate_mem((void**) &output, img->num_components * img->num_components * sizeof(type_data));
//	checkCUDAError("after calulation cov matrix");
//	println_var(INFO, "after calculation cov matrix");

//	cublasStatus status;
//	status = cublasInit();
//	if (status != CUBLAS_STATUS_SUCCESS) {
//		println_var(INFO, "ERROR Cublas Init Error with status %d", status);
//		checkCUDAError("cublas init error");
//		return;
//	}
//	checkCUDAError("after cublas init");
//	println_var(INFO, "after cublas init");

#ifdef PART_TIME
	cudaThreadSynchronize();
	printf("calc cov mat: %ld\n", stop_measure(start_calc_cov_mat));
	long int start_calc_eigen;
	start_calc_eigen = start_measure();
#endif

	/* calculating eigenvectors and eigenvalues */
   
	clock_t gs_start = clock();
#ifdef MCT_CALC_GS
	gram_schmidt(img->num_components, output, covMatrix_d, eValues, param->param_mct_klt_iterations, param->param_mct_klt_err);
#endif
	//println_var(INFO, "Time of GS %f", ((double)clock() - gs_start)/ CLOCKS_PER_SEC);
	
#ifdef PART_TIME
	cudaThreadSynchronize();
	printf("calc eigen: %ld\n", stop_measure(start_calc_eigen));
	long int start_look_drop_comp;
	start_look_drop_comp = start_measure();
#endif

	/* looking for components to drop */

//	checkCUDAError("after gs");
//	println_var(INFO, "after gs");
	int i,odciecie = 0;
	type_data max = eValues[0];
	for(i=0; i<img->num_components; ++i) {
//		printf("%f\n", eValues[i]);
		eValues[i] = eValues[i]/max;
//		printf("%f\n", eValues[i]);
	}
	i = img->num_components - 1;
	odciecie = 0;
	while(eValues[i] <= param->param_mct_klt_border_eigenvalue) {
		++odciecie;
		--i;
		if(i==0)
			break;
	}

#ifndef MCT_DROP_COMPONENTS
	odciecie = 0;
#endif

#ifdef PART_TIME
	cudaThreadSynchronize();
	printf("calc look for comp to drop: %ld\n", stop_measure(start_look_drop_comp));
	long int start_prep_trans_mat;
	start_prep_trans_mat = start_measure();
#endif

	/* preparing transform matrix */

//	checkCUDAError("before prepare transform matrix");
//	println_var(INFO, "before prepare transform matrix");
	type_data* transform;
	type_data* transform_d;
	cuda_h_allocate_mem((void **) &transform, img->num_components * (img->num_components - odciecie) * sizeof(type_data));
	cuda_d_allocate_mem((void **) &transform_d, img->num_components * (img->num_components - odciecie) * sizeof(type_data));
	for(i=0; i<img->num_components; ++i) {
		for(int j=0; j<img->num_components - odciecie; ++j) {
			transform[i * (img->num_components - odciecie) + j] = output[i * img->num_components + j];
		}
	}

	cuda_memcpy_htd((void *) transform, (void *) transform_d, img->num_components * (img->num_components - odciecie) * sizeof(type_data));


	cuda_d_allocate_mem((void **)&transform_d, img->num_components * img->num_components * sizeof(type_data));
	cuda_memcpy_htd((void*)output, transform_d, img->num_components * img->num_components * sizeof(type_data));

#ifdef PART_TIME
	cudaThreadSynchronize();
	printf("prep trans mat: %ld\n", stop_measure(start_prep_trans_mat));
	long int start_adj_data;
	start_adj_data = start_measure();
#endif

#ifdef PART_TIME
	long int start_adj_data_1;
	start_adj_data_1 = start_measure();
#endif
	/* adjusting data */

//	checkCUDAError("before data adjust");
#ifdef PART_TIME
	cudaThreadSynchronize();
	printf("adj data1: %ld\n", stop_measure(start_adj_data_1));
#endif
#ifdef PART_TIME
	long int start_adj_data_2;
	start_adj_data_2 = start_measure();
#endif
//	println_var(INFO, "before data adjust");
#ifdef MCT_TRANSFORM_DATA
	mct_transform_new(img, transform_d, data_pd, odciecie);
#endif
#ifdef PART_TIME
	cudaThreadSynchronize();
	printf("adj data2: %ld\n", stop_measure(start_adj_data_2));
#endif
#ifdef PART_TIME
	long int start_adj_data_3;
	start_adj_data_3 = start_measure();
#endif
//	checkCUDAError("after data adjust");
//	println_var(INFO, "after data adjust");

/*	status = cublasShutdown();
	if (status != CUBLAS_STATUS_SUCCESS) {
		println_var(INFO, "ERROR Cublas Shutdown Error with status %d", status);
		checkCUDAError("cublas shutdown error");
		return;
	}*/

//	checkCUDAError("after cublas shutdown");
//	println_var(INFO, "after cublas shutdown");
#ifdef PART_TIME
	cudaThreadSynchronize();
	printf("adj data3: %ld\n", stop_measure(start_adj_data_3));
#endif
#ifdef PART_TIME
	cudaThreadSynchronize();
	printf("adj data: %ld\n", stop_measure(start_adj_data));
	long int start_drop_comp;
	start_drop_comp = start_measure();
#endif

	/* drops component data */
	
	for(int i=img->num_components - odciecie; i < img->num_components; ++i) {
		cuda_d_free(data_p[i]);
	}
//	checkCUDAError("after component dropping");
//	println_var(INFO, "after component dropping");

	/* crating segments for decoding purposes */

	type_mct* mct_matrix = (type_mct*)calloc(1,sizeof(type_mct));
	type_mct* mct_offset = (type_mct*)calloc(1,sizeof(type_mct));
	
	mct_offset->index = 0;
	mct_offset->type = MCT_DECORRELATION_OFFSET;
	mct_offset->element_type = MCT_32BIT_FLOAT;
	mct_offset->length = img->num_components;
	mct_offset->data = (uint8_t*)means;

	img->mct_data->mcts_count[MCT_DECORRELATION_OFFSET] = 1;
	img->mct_data->mcts[MCT_DECORRELATION_OFFSET] = mct_offset;

	mct_matrix->index = 0;
	mct_matrix->type = MCT_DECORRELATION_TRANSFORMATION;
	mct_matrix->element_type = MCT_32BIT_FLOAT;
	mct_matrix->length = img->num_components * (img->num_components - odciecie);
	mct_matrix->data = (uint8_t*)transform; // TODO verify if pointer is valid (code reading)

	img->mct_data->mcts_count[MCT_DECORRELATION_TRANSFORMATION] = 1;
	img->mct_data->mcts[MCT_DECORRELATION_TRANSFORMATION] = mct_matrix;
	
	type_mcc* mcc = (type_mcc*)calloc(1, sizeof(type_mcc));
	mcc->index = 0;
	mcc->count = 1;
	mcc->data = (type_mcc_data*)calloc(1,sizeof(type_mcc_data));
	
	mcc->data->type = MCC_MATRIX_BASED;
	mcc->data->input_count = img->num_components - odciecie;
	mcc->data->input_component_type = img->num_components>0x3FFF?MCT_16BIT_INT:MCT_8BIT_INT;
	mcc->data->output_count = img->num_components;
	mcc->data->output_component_type = img->num_components>0x3FFF?MCT_16BIT_INT:MCT_8BIT_INT;
	mcc->data->decorrelation_transform_matrix = 0;
	mcc->data->deccorelation_transform_offset = 0;
	
	if(img->num_components > 0x3FFF) {
		mcc->data->input_components = (uint8_t*)calloc(img->num_components - odciecie, sizeof(uint16_t));
		mcc->data->output_components = (uint8_t*)calloc(img->num_components, sizeof(uint16_t));
		uint16_t* ins = (uint16_t*)mcc->data->input_components;
		uint16_t* outs = (uint16_t*)mcc->data->output_components;
		for(i=0; i<img->num_components; ++i) {
			outs[i] = i;
			if(i < img->num_components - odciecie) {
				ins[i] = i;
			}
		}
	} else {
		mcc->data->input_components = (uint8_t*)calloc(img->num_components - odciecie, sizeof(uint8_t));
		mcc->data->output_components = (uint8_t*)calloc(img->num_components, sizeof(uint8_t));
		for(i=0; i<img->num_components; ++i) {
			mcc->data->output_components[i] = i;
			if(i < img->num_components - odciecie) {
				mcc->data->input_components[i] = i;
			}
		}
	}
	
	img->mct_data->mccs_count = 1;
	img->mct_data->mccs = mcc;

	cuda_h_free(means);
	cuda_h_free((void*)data_p);
	cuda_d_free((void*)data_pd);

#ifdef PART_TIME
	cudaThreadSynchronize();
	printf("drop comp data: %ld\n", stop_measure(start_drop_comp));
#endif


	img->num_components -= odciecie; 
	//println_var(INFO, "Time of MCT processing %f", ((double)clock() - start)/ CLOCKS_PER_SEC);
//	printf("%ld\n", stop_measure(start_klt));
//	println_var(INFO, "MCT %d components droped", odciecie);
//	checkCUDAError("after MCT");
//	println_var(INFO, "after MCT");
}

void decode_tile_mcc(type_mcc_data* mcc, type_tile* tile, type_image* img) {

	clock_t start = clock();
	type_mct* mct_matrix;
	type_mct* mct_offset;

	int i;
	for(i=0; img->mct_data->mcts_count[MCT_DECORRELATION_TRANSFORMATION]; ++i) {
		mct_matrix = img->mct_data->mcts[MCT_DECORRELATION_TRANSFORMATION];
		if(mct_matrix->index == mcc->decorrelation_transform_matrix)
			break;
	}

	for(i=0; img->mct_data->mcts_count[MCT_DECORRELATION_OFFSET]; ++i) {
		mct_offset = img->mct_data->mcts[MCT_DECORRELATION_OFFSET];
		if(mct_offset->index == mcc->deccorelation_transform_offset)
			break;
	}

	int j=0, k=0;
	type_data* matrix = (type_data*)my_malloc(sizeof(type_data)*mct_matrix->length);
	type_data* off  = (type_data*)my_malloc(sizeof(type_data)*mct_offset->length);
	type_data* matrix_d;
	type_data* off_d;
	cudaMalloc(&matrix_d, sizeof(type_data)*mct_matrix->length);
	cudaMalloc(&off_d, sizeof(type_data)*mct_offset->length);

	for(i=0; i<mct_matrix->length; ++i) {
		if(++j==mcc->output_count) {
			j=0;
			++k;
		}
		switch(mct_matrix->element_type) {
			case MCT_8BIT_INT:
				matrix[k * mcc->output_count + j] = (type_data) ((uint8_t*)mct_matrix->data)[i]; break;
			case MCT_16BIT_INT:
				matrix[k * mcc->output_count + j] = (type_data)((uint16_t*)mct_matrix->data)[i]; break;
			case MCT_32BIT_FLOAT:
				matrix[k * mcc->output_count + j] = (type_data)((float*)mct_matrix->data)[i]; break;
			case MCT_64BIT_DOUBLE:
				matrix[k * mcc->output_count + j] = (type_data)((double*)mct_matrix->data)[i]; break;
		}
	}

	for(k=0; k<mcc->output_count; ++k) {
		switch(mct_offset->element_type) {
			case MCT_8BIT_INT:
				off[k] = (type_data)((uint8_t *)mct_offset->data)[k]; break;
			case MCT_16BIT_INT:
				off[k] = (type_data)((uint16_t *)mct_offset->data)[k]; break;
			case MCT_32BIT_FLOAT:
				off[k] = (type_data)((float *)mct_offset->data)[k]; break;
			case MCT_64BIT_DOUBLE:
				off[k] = (type_data)((double *)mct_offset->data)[k]; break;
		}
	}

	cudaMemcpy(matrix_d, matrix, sizeof(type_data)*mct_matrix->length, cudaMemcpyHostToDevice);
	cudaMemcpy(off_d, off, sizeof(type_data)*mct_offset->length, cudaMemcpyHostToDevice);

	type_data* data_d;
	type_data* inter_d;
	cudaMalloc(&inter_d, sizeof(type_data)*mcc->input_count);
	cudaMalloc(&data_d, sizeof(type_data)*mcc->output_count);

	int delta = mcc->output_count - mcc->input_count;
	img->num_components += delta;
	println_var(INFO, "%d new components", delta);
	type_tile_comp *tile_comp;
	
	for(i=0; i<img->num_tiles; ++i) {
		// alloc components dropt in encoding process
	
		img->tile[i].tile_comp = (type_tile_comp*)realloc((void*)img->tile[i].tile_comp, sizeof(type_tile_comp) * img->num_components);
		if(img->tile[i].tile_comp == NULL) {
			println_var(INFO, "MEMORY REALLOCATION ERROR CANNOT ALLOCATE COMPONENTS!");
			exit(EXIT_FAILURE);
		}

		for(j=img->num_components-delta; j<img->num_components; ++j) {
			tile_comp = &(img->tile[i].tile_comp[j]);
			tile_comp->width = img->tile[i].width;
			tile_comp->height = img->tile[i].height;
			cuda_d_allocate_mem((void **) &(tile_comp->img_data_d), tile_comp->width * tile_comp->height * sizeof(type_data));
			// other fields seems to not be in use for saving image so we can ignore them ...
		}
	}


	type_data** components_out_p = (type_data**)calloc(mcc->output_count,sizeof(type_data*));
	type_data** components_out_pd;
	cudaMalloc(&components_out_pd, mcc->output_count*sizeof(type_data*));

	for(k=0; k < mcc->output_count; ++k) {
		components_out_p[k] = img->tile[0].tile_comp[mcc->output_components[k]].img_data_d;
	}
	cudaMemcpy(components_out_pd, components_out_p, sizeof(type_data*)*mcc->output_count, cudaMemcpyHostToDevice);

	type_data** components_in_p = (type_data**)calloc(mcc->input_count,sizeof(type_data*));
	type_data** components_in_pd;
	cudaMalloc(&components_in_pd, mcc->input_count*sizeof(type_data*));

	for(k=0; k < mcc->input_count; ++k) {
		components_in_p[k] = img->tile[0].tile_comp[mcc->input_components[k]].img_data_d;
	}
	cudaMemcpy(components_in_pd, components_in_p, sizeof(type_data*)*mcc->input_count, cudaMemcpyHostToDevice);

	for(j=0; j<img->tile[0].width * img->tile[0].height; ++j) {
	cudaThreadSynchronize();
		readSampleSimple<<<1,mcc->input_count>>>(components_in_pd, j, inter_d);
	cudaThreadSynchronize();
		adjust_pca_data_mm(matrix_d, 0, inter_d, data_d, mcc->input_count, mcc->output_count);
	cudaThreadSynchronize();
		writeSampleWithSum<<<1,mcc->output_count>>>(components_out_pd, j, data_d, off_d);
	}
	println_var(INFO, "Time of MCT processing %f", ((double)clock() - start)/ CLOCKS_PER_SEC);
}

void decode_klt(type_image *img) {
	int i,j,k;
	type_mcc* mcc;
	type_mcc_data* mcc_data;
	for(i=0; i<img->mct_data->mccs_count; ++i) {
		mcc = &img->mct_data->mccs[i];
		for(j=0; j<mcc->count;++j) {
			mcc_data = &mcc->data[j];
			if(mcc_data->type == MCC_MATRIX_BASED) {
				for(k=0; k<img->num_tiles; ++k) {
					// TODO: check for tile scecific modifications of MCC (not supported)
					decode_tile_mcc(mcc_data, &img->tile[k],img);
				}
			}
		}
	}
}

