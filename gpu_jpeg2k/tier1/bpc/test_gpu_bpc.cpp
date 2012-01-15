/**
 * @file test_gpu_bpc.cpp
 *
 * @author Milosz Ciznicki
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

extern "C" {
#include "../../misc/memory_management.cuh"
#include "../../print_info/print_info.h"
#include "../ebcot/mqc/mqc_data.h"
}

#include "gpu_bpc.h"
#include "test_gpu_bpc.h"

typedef struct {
	unsigned char d;
	unsigned char cx;
	unsigned int tid;
}cxd_pair;

void binary_printf(unsigned int in)
{
	for(int i = 0; i < 32; i++) {
		if((in >> (31 - i)) & 1)
			printf("1");
		else
			printf("0");
		if(i % 8 == 7)
			printf(" ");
	}

	printf("\n");
}

int get_exp_subband_gain(int orient)
{
	return (orient & 1) + ((orient >> 1) & 1);
}

void print_cxd(unsigned int pairs) {
	unsigned char counter = pairs & CXD_COUNTER;

	printf("%x\n", pairs);
	for(int i = 0; i < counter; ++i) {
		unsigned char d = (pairs >> (D1_BITPOS - i * 6)) & 0x1;
		unsigned char cx = (pairs >> (CX1_BITPOS - i * 6)) & 0x1f;
		printf("- %d %d\n", d, cx);
	}
}

void encode_bpc_test(const char *file_name) {
	struct mqc_data* mqc_data = mqc_data_create_from_image(file_name);
	if (mqc_data == 0) {
		std::cerr << "Can't receive data from openjpeg: " << std::endl;
		return;
	}

	// Initialize CUDA
	cudaError cuerr = cudaSuccess;

	int codeBlocks = mqc_data->cblk_count;
	// maximum 6 CX, D pairs per coeff in codeblock
	int maxOutLength = /*mqc_data->cblks[0]->w * mqc_data->cblks[0]->h * 6*/4096*10;

	printf("codeBlocks %d %d\n", codeBlocks, mqc_data->cblks[0]->cxd_count);

	unsigned int *d_cxd_pairs;
//	GPU_JPEG2K::CoefficientState *d_stBuffors;

	CodeBlockAdditionalInfo *h_infos = (CodeBlockAdditionalInfo *) malloc(sizeof(CodeBlockAdditionalInfo) * codeBlocks);
	CodeBlockAdditionalInfo *d_infos;

	cuda_d_allocate_mem((void **) &d_cxd_pairs, sizeof(unsigned int) * codeBlocks * maxOutLength);
	cuda_d_allocate_mem((void **) &d_infos, sizeof(CodeBlockAdditionalInfo) * codeBlocks);

	int magconOffset = 0;

	printf("Before loop\n");

	for (int i = 0; i < codeBlocks; i++) {
		struct mqc_data_cblk *cblk = mqc_data->cblks[i];
		h_infos[i].width = cblk->w;
		h_infos[i].height = cblk->h;
		h_infos[i].nominalWidth = cblk->w;
		h_infos[i].stripeNo = (int) ceil(cblk->h / 4.0f);
		cblk->subband--;
		h_infos[i].subband = cblk->subband > 0 ? cblk->subband : 0;
		h_infos[i].magconOffset = magconOffset + cblk->w;
		h_infos[i].magbits = /*9 + get_exp_subband_gain(cblk->subband)*/cblk->magbits;
		int max = 0;
		for(int j = 0; j < cblk->w; ++j) {
                        for(int k = 0; k < cblk->h; ++k) {
//				if(cblk->coefficients[k * cblk->w + j] < 0)
//					printf("-");
//				binary_printf(abs(cblk->coefficients[k * cblk->w + j]));
//                                cblk->coefficients[k * cblk->w + j] <<= (31 - 6 - (cblk->magbits));
                                if(cblk->coefficients[k * cblk->w + j] > max)
                                        max = cblk->coefficients[k * cblk->w + j];
                        }
                }
//		binary_printf(max);
//		binary_printf(cblk->coefficients[0]);
		max = 0;
		for(int j = 0; j < cblk->w; ++j) {
			for(int k = 0; k < cblk->h; ++k) {
				int cache_value = (cblk->coefficients[k * cblk->w + j]) << (31 - 6 - h_infos[i].magbits);
				cblk->coefficients[k * cblk->w + j] = cache_value < 0 ? (1 << 31) | (-cache_value) : cache_value;
//				binary_printf(cblk->coefficients[k * cblk->w + j]);
//				printf("%x\n", cache_value < 0 ? (1 << 31) | (-cache_value) : cache_value);
				if(cblk->coefficients[k * cblk->w + j] > max)
					max = cblk->coefficients[k * cblk->w + j];
			}
		}
//		binary_printf(max);
//		binary_printf(cblk->coefficients[0]);
		cuda_d_allocate_mem((void **)&(h_infos[i].coefficients), h_infos[i].width * h_infos[i].height * sizeof(int));
		cuda_memcpy_htd(cblk->coefficients, h_infos[i].coefficients, h_infos[i].width * h_infos[i].height * sizeof(int));
		//h_infos[i].coefficients = cblk->coefficients;
		h_infos[i].compType = cblk->compType;
		h_infos[i].stepSize = cblk->stepSize;
		switch(cblk->dwtLevel) {
		case 4:
		case 3:
			h_infos[i].dwtLevel = 4;
			break;
		case 2:
			h_infos[i].dwtLevel = 3;
			break;
		case 1:
			h_infos[i].dwtLevel = 2;
			break;
		case 0:
			h_infos[i].dwtLevel = 1;
			break;
		}

		magconOffset += h_infos[i].width * (h_infos[i].stripeNo + 2);

		h_infos[i].MSB = 0;

//		printf("%d %d %d %d %d %d %d %d %d %f\n", h_infos[i].width, h_infos[i].height, h_infos[i].nominalWidth,
//				h_infos[i].stripeNo, h_infos[i].subband, h_infos[i].magconOffset, h_infos[i].magbits,
//				h_infos[i].compType, h_infos[i].dwtLevel, h_infos[i].stepSize);
	}

//	binary_printf(mqc_data->cblks[0]->coefficients[0]);

//	cuda_d_allocate_mem((void **) &d_stBuffors, sizeof(GPU_JPEG2K::CoefficientState) * magconOffset);
//	CHECK_ERRORS(cudaMemset((void *) d_stBuffors, 0, sizeof(GPU_JPEG2K::CoefficientState) * magconOffset));

	cuda_memcpy_htd(h_infos, d_infos, sizeof(CodeBlockAdditionalInfo) * codeBlocks);

	printf("\n");

	int w = mqc_data->cblks[0]->w;
	int h = mqc_data->cblks[0]->h;
	int max_dim = (w > h) ? w : h;

	dim3 gridDim(1,1,1);
	dim3 blockDim(max_dim, max_dim, 1);

	launch_bpc_encode(gridDim, blockDim, d_infos, d_cxd_pairs);

	unsigned int *h_cxd_pairs = NULL;
	cuda_h_allocate_mem((void **) &h_cxd_pairs, sizeof(unsigned int) * codeBlocks * maxOutLength);
	cuda_memcpy_dth(d_cxd_pairs, h_cxd_pairs, sizeof(unsigned int) * codeBlocks * maxOutLength);

	cuda_memcpy_dth(d_infos, h_infos, sizeof(CodeBlockAdditionalInfo) * codeBlocks);

	printf("MSB %d\n", h_infos[0].MSB);

	int pairs_count = 0;
	for (int i = 0; i < codeBlocks; ++i) {
		for(int j = 0; j < w*h; ++j) {
			pairs_count += h_cxd_pairs[i * maxOutLength + j] & CXD_COUNTER;
		}
	}

	cxd_pair *cxd_pairs = (cxd_pair *) malloc(sizeof(cxd_pair) * pairs_count);

	int curr_pair = 0;
	for (int i = 0; i < codeBlocks; ++i) {
		for(int j = 0; j < w*h; ++j) {
			unsigned char counter = h_cxd_pairs[i * maxOutLength + j] & CXD_COUNTER;

			for(int k = 0; k < counter; ++k) {
				unsigned char d = (h_cxd_pairs[i * maxOutLength + j] >> (D1_BITPOS - k * 6)) & 0x1;
				unsigned char cx = (h_cxd_pairs[i * maxOutLength + j] >> (CX1_BITPOS - k * 6)) & 0x1f;
//				if((i * maxOutLength + j) == 47) {
//					printf("%x\n", h_cxd_pairs[i * maxOutLength + j]);
//				}
				cxd_pairs[curr_pair].d = d;
				cxd_pairs[curr_pair].cx = cx;
				cxd_pairs[curr_pair].tid = i * maxOutLength + j;
				++curr_pair;
			}
		}
	}

	printf("\n\n\n");
	for (int i = 0; i < codeBlocks; ++i) {
		struct mqc_data_cblk *cblk = mqc_data->cblks[i];
		for(int j = 0; j < /*cblk->cxd_count*/pairs_count; ++j) {
			//if((cblk->cxds[j].d != ((h_cxd_pairs[i * maxOutLength + j]&(1<<5)) >> 5)) || (cblk->cxds[j].cx != (h_cxd_pairs[i * maxOutLength + j]&0x1f))) {
			if((cblk->cxds[j].cx != cxd_pairs[j].cx) || (cblk->cxds[j].d != cxd_pairs[j].d)) {
				printf("%d) + %d %d", j, cblk->cxds[j].d, cblk->cxds[j].cx);
				printf("	- %d %d	%d\n", cxd_pairs[j].d, cxd_pairs[j].cx, cxd_pairs[j].tid);
			}
			//}
		}
		/*if (cblk->cxd_count != h_infos[i].length) {
			std::cerr << cblk->cxd_count << " != " << h_infos[i].length << std::endl;
			for(int j = 0; j < h_infos[i].length; ++j) {
//				if(cblk->cxds[j] != h_cxd_pairs[i * maxOutLength + j]) {
					std::cerr << i <<  ") ";
					if(j < cblk->cxd_count)
						binary_printf((cblk->cxds[j].d << 5) | cblk->cxds[j].cx);
					std::cerr << "   ";
					binary_printf(h_cxd_pairs[i * maxOutLength + j]);
					std::cerr << std::endl;
//				}
			}
		}*/
	}
}
