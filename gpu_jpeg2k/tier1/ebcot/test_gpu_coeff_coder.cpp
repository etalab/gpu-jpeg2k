/**
 * @file test_gpu_coeff_coder.cpp
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
#include "mqc/mqc_data.h"
}

#include "gpu_coder.h"
#include "gpu_coeff_coder2.cuh"

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

void encode_tasks_test() {
	char file_name[128];
	sprintf(file_name, "/home/miloszc/Projects/images/rgb8bit/flower_foveon.ppm\0");
	struct mqc_data* mqc_data = mqc_data_create_from_image(file_name);
	if (mqc_data == 0) {
		std::cerr << "Can't receive data from openjpeg: " << std::endl;
		return;
	}

	// Initialize CUDA
	cudaError cuerr = cudaSuccess;

	int codeBlocks = mqc_data->cblk_count;
	// maximum 6 CX, D pairs per coeff in codeblock
	int maxOutLength = /*mqc_data->cblks[0]->w * mqc_data->cblks[0]->h * 6*/4096*6;

	printf("codeBlocks %d %d\n", codeBlocks, maxOutLength);

	byte *d_outbuf;
	byte *d_cxd_pairs;
	GPU_JPEG2K::CoefficientState *d_stBuffors;

	CodeBlockAdditionalInfo *h_infos = (CodeBlockAdditionalInfo *) malloc(sizeof(CodeBlockAdditionalInfo) * codeBlocks);
	CodeBlockAdditionalInfo *d_infos;

	cuda_d_allocate_mem((void **) &d_outbuf, sizeof(byte) * codeBlocks * maxOutLength);
	cuda_d_allocate_mem((void **) &d_cxd_pairs, sizeof(byte) * codeBlocks * maxOutLength);
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
//				cblk->coefficients[k * cblk->w + j] <<= (31 - 6 - h_infos[i].magbits);
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

//		printf("%d %d %d %d %d %d %d %d %d %f\n", h_infos[i].width, h_infos[i].height, h_infos[i].nominalWidth,
//				h_infos[i].stripeNo, h_infos[i].subband, h_infos[i].magconOffset, h_infos[i].magbits,
//				h_infos[i].compType, h_infos[i].dwtLevel, h_infos[i].stepSize);
	}

	cuda_d_allocate_mem((void **) &d_stBuffors, sizeof(GPU_JPEG2K::CoefficientState) * magconOffset);
	CHECK_ERRORS(cudaMemset((void *) d_stBuffors, 0, sizeof(GPU_JPEG2K::CoefficientState) * magconOffset));

	cuda_memcpy_htd(h_infos, d_infos, sizeof(CodeBlockAdditionalInfo) * codeBlocks);

	CHECK_ERRORS(GPU_JPEG2K::launch_encode((int) ceil((float) codeBlocks / THREADS), THREADS, d_stBuffors, d_outbuf, d_cxd_pairs, maxOutLength, d_infos, codeBlocks));

	byte *h_cxd_pairs = NULL;
	cuda_h_allocate_mem((void **) &h_cxd_pairs, sizeof(byte) * codeBlocks * maxOutLength);
	cuda_memcpy_dth(d_cxd_pairs, h_cxd_pairs, sizeof(byte) * codeBlocks * maxOutLength);

	cuda_memcpy_dth(d_infos, h_infos, sizeof(CodeBlockAdditionalInfo) * codeBlocks);

	for (int i = 0; i < codeBlocks; ++i) {
		struct mqc_data_cblk *cblk = mqc_data->cblks[i];
		if (cblk->cxd_count != h_infos[i].length) {
			std::cerr << cblk->cxd_count << " != " << h_infos[i].length << std::endl;
		}
	}
}
