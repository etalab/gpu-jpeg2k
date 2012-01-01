/**
 * @file gpu_coeff_coder.cpp
 *
 * @brief Coefficients coder.
 */

extern "C" {
	#include "gpu_coder.h"
	#include "../../misc/memory_management.cuh"
	#include "../../print_info/print_info.h"
}

#include <iostream>
#include <string>
#include <fstream>

//#include <libxml++/libxml++.h>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include <list>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "gpu_coeff_coder2.cuh"
#include "coeff_coder_pcrd.cuh"
#include "mqc/mqc_wrapper.h"

void print_cdx(EntropyCodingTaskInfo *infos, int codeBlocks) {
	for(int i = 0; i < codeBlocks; i++)
	{
		printf("%d) %d\n", i, infos[i].length);
	}
}

float gpuEncode(EntropyCodingTaskInfo *infos, int count, int targetSize)
{
	int codeBlocks = count;
	int maxOutLength = MAX_CODESTREAM_SIZE;

	int n = 0;
	for(int i = 0; i < codeBlocks; i++)
		n += infos[i].width * infos[i].height;

	byte *d_outbuf;
	byte *d_cxd_pairs;
	GPU_JPEG2K::CoefficientState *d_stBuffors;

	CodeBlockAdditionalInfo *h_infos = (CodeBlockAdditionalInfo *) malloc(sizeof(CodeBlockAdditionalInfo) * codeBlocks);
	CodeBlockAdditionalInfo *d_infos;

	cuda_d_allocate_mem((void **) &d_cxd_pairs, sizeof(byte) * codeBlocks * maxOutLength);
	cuda_d_allocate_mem((void **) &d_infos, sizeof(CodeBlockAdditionalInfo) * codeBlocks);

	int magconOffset = 0;

	for(int i = 0; i < codeBlocks; i++)
	{
		h_infos[i].width = infos[i].width;
		h_infos[i].height = infos[i].height;
		h_infos[i].nominalWidth = infos[i].nominalWidth;
		h_infos[i].stripeNo = (int) ceil(infos[i].height / 4.0f);
		h_infos[i].subband = infos[i].subband;
		h_infos[i].magconOffset = magconOffset + infos[i].width;
		h_infos[i].magbits = infos[i].magbits;
		h_infos[i].coefficients = infos[i].coefficients;
		h_infos[i].compType = infos[i].compType;
		h_infos[i].dwtLevel = infos[i].dwtLevel;
		h_infos[i].stepSize = infos[i].stepSize;

		magconOffset += h_infos[i].width * (h_infos[i].stripeNo + 2);
	}

	cuda_d_allocate_mem((void **) &d_stBuffors, sizeof(GPU_JPEG2K::CoefficientState) * magconOffset);
	CHECK_ERRORS(cudaMemset((void *) d_stBuffors, 0, sizeof(GPU_JPEG2K::CoefficientState) * magconOffset));

	cuda_memcpy_htd(h_infos, d_infos, sizeof(CodeBlockAdditionalInfo) * codeBlocks);

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start, 0);

	if(targetSize == 0)
	{
		//printf("No pcrd\n");
		CHECK_ERRORS(GPU_JPEG2K::launch_encode((int) ceil((float) codeBlocks / THREADS), THREADS, d_stBuffors, d_cxd_pairs, maxOutLength, d_infos, codeBlocks));
	}
	else
	{
		printf("Pcrd\n");
		CHECK_ERRORS(GPU_JPEG2K::launch_encode_pcrd((int) ceil((float) codeBlocks / THREADS), THREADS, d_stBuffors, maxOutLength, d_infos, codeBlocks, targetSize));
	}

	cudaEventRecord(end, 0);

	cuda_memcpy_dth(d_infos, h_infos, sizeof(CodeBlockAdditionalInfo) * codeBlocks);

	mqc_gpu_encode(infos, h_infos, codeBlocks, d_cxd_pairs, maxOutLength);

	for(int i = 0; i < codeBlocks; i++)
	{
		infos[i].significantBits = h_infos[i].significantBits;
		infos[i].codingPasses = h_infos[i].codingPasses;

		/*if(h_infos[i].length > 0)
		{
			infos[i].length = h_infos[i].length;

			int len = h_infos[i].length;

			infos[i].codeStream = (byte *) malloc(sizeof(byte) * len);
			cuda_memcpy_dth(d_outbuf + i * maxOutLength, infos[i].codeStream, sizeof(byte) * len);
		}
		else
		{
			infos[i].length = 0;
			infos[i].codeStream = NULL;
		}*/
	}

//	print_cdx(infos, codeBlocks);

	cuda_d_free(d_stBuffors);
	cuda_d_free(d_infos);

	free(h_infos);

	float elapsed = 0.0f;
	cudaEventElapsedTime(&elapsed, start, end);
	
	return elapsed;
}

float gpuDecode(EntropyCodingTaskInfo *infos, int count)
{
	int codeBlocks = count;
	int maxOutLength = MAX_CODESTREAM_SIZE;

	int n = 0;
	for(int i = 0; i < codeBlocks; i++)
		n += infos[i].width * infos[i].height;

	byte *d_inbuf;
	GPU_JPEG2K::CoefficientState *d_stBuffors;

	CodeBlockAdditionalInfo *h_infos = (CodeBlockAdditionalInfo *) malloc(sizeof(CodeBlockAdditionalInfo) * codeBlocks);
	CodeBlockAdditionalInfo *d_infos;

	cuda_d_allocate_mem((void **) &d_inbuf, sizeof(byte) * codeBlocks * maxOutLength);
	cuda_d_allocate_mem((void **) &d_infos, sizeof(CodeBlockAdditionalInfo) * codeBlocks);

	int magconOffset = 0;

	for(int i = 0; i < codeBlocks; i++)
	{
		h_infos[i].width = infos[i].width;
		h_infos[i].height = infos[i].height;
		h_infos[i].nominalWidth = infos[i].nominalWidth;
		h_infos[i].stripeNo = (int) ceil(infos[i].height / 4.0f);
		h_infos[i].subband = infos[i].subband;
		h_infos[i].magconOffset = magconOffset + infos[i].width;
		h_infos[i].magbits = infos[i].magbits;
		h_infos[i].length = infos[i].length;
		h_infos[i].significantBits = infos[i].significantBits;

		cuda_d_allocate_mem((void **) &(h_infos[i].coefficients), sizeof(int) * infos[i].nominalWidth * infos[i].nominalHeight);
		infos[i].coefficients = h_infos[i].coefficients;

		cuda_memcpy_htd(infos[i].codeStream, (void *) (d_inbuf + i * maxOutLength), sizeof(byte) * infos[i].length);

		magconOffset += h_infos[i].width * (h_infos[i].stripeNo + 2);
	}

	cuda_d_allocate_mem((void **) &d_stBuffors, sizeof(GPU_JPEG2K::CoefficientState) * magconOffset);
	CHECK_ERRORS(cudaMemset((void *) d_stBuffors, 0, sizeof(GPU_JPEG2K::CoefficientState) * magconOffset));

	cuda_memcpy_htd(h_infos, d_infos, sizeof(CodeBlockAdditionalInfo) * codeBlocks);

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start, 0);

	CHECK_ERRORS(GPU_JPEG2K::launch_decode((int) ceil((float) codeBlocks / THREADS), THREADS, d_stBuffors, d_inbuf, maxOutLength, d_infos, codeBlocks));

	cudaEventRecord(end, 0);

	cuda_d_free(d_inbuf);
	cuda_d_free(d_stBuffors);
	cuda_d_free(d_infos);

	free(h_infos);

	float elapsed = 0.0f;
	cudaEventElapsedTime(&elapsed, start, end);
	
	return elapsed;
}

void convert_to_task(EntropyCodingTaskInfo &task, const type_codeblock &cblk)
{
	task.coefficients = cblk.data_d;

	switch(cblk.parent_sb->orient)
	{
	case LL:
	case LH:
		task.subband = 0;
		break;
	case HL:
		task.subband = 1;
		break;
	case HH:
		task.subband = 2;
		break;
	}

	task.width = cblk.width;
	task.height = cblk.height;

	task.nominalWidth = cblk.parent_sb->parent_res_lvl->parent_tile_comp->cblk_w;
	task.nominalHeight = cblk.parent_sb->parent_res_lvl->parent_tile_comp->cblk_h;

	task.magbits = cblk.parent_sb->mag_bits;
	task.compType = cblk.parent_sb->parent_res_lvl->parent_tile_comp->parent_tile->parent_img->wavelet_type;
	task.dwtLevel = cblk.parent_sb->parent_res_lvl->dec_lvl_no;
	task.stepSize = cblk.parent_sb->step_size;
}

void extract_cblks(type_tile *tile, std::list<type_codeblock *> &out_cblks)
{
	for(int i = 0; i < tile->parent_img->num_components; i++)
	{
		type_tile_comp *tile_comp = &(tile->tile_comp[i]);
		for(int j = 0; j < tile_comp->num_rlvls; j++)
		{
			type_res_lvl *res_lvl = &(tile_comp->res_lvls[j]);
			for(int k = 0; k < res_lvl->num_subbands; k++)
			{
				type_subband *sb = &(res_lvl->subbands[k]);
				for(int l = 0; l < sb->num_cblks; l++)
					out_cblks.push_back(&(sb->cblks[l]));
			}
		}
	}
}

void binary_fprintf(FILE *file, unsigned int in)
{
	for(int i = 0; i < 32; i++)
		if((in >> (31 - i)) & 1)
			fprintf(file, "1");
		else
			fprintf(file, "0");

	fprintf(file, "\n");
}

void encode_tasks_serial(type_tile *tile) {
	type_coding_param *coding_params = tile->parent_img->coding_param;

	std::list<type_codeblock *> cblks;
	extract_cblks(tile, cblks);

	EntropyCodingTaskInfo *tasks = (EntropyCodingTaskInfo *) malloc(sizeof(EntropyCodingTaskInfo) * cblks.size());

	std::list<type_codeblock *>::iterator ii = cblks.begin();

	int num_tasks = 0;
	for(; ii != cblks.end(); ++ii)
		convert_to_task(tasks[num_tasks++], *(*ii));

//	printf("%d\n", num_tasks);

	float t = gpuEncode(tasks, num_tasks, coding_params->target_size);

//	printf("kernel consumption: %f\n", t);

	ii = cblks.begin();

	for(int i = 0; i < num_tasks; i++, ++ii)
	{
		(*ii)->codestream = tasks[i].codeStream;
		(*ii)->length = tasks[i].length;
		(*ii)->significant_bits = tasks[i].significantBits;
		(*ii)->num_coding_passes = tasks[i].codingPasses;
	}

	free(tasks);
}

void encode_tile(type_tile *tile)
{
//	println_start(INFO);

//	start_measure();

	encode_tasks_serial(tile);

//	stop_measure(INFO);

//	println_end(INFO);
}

void convert_to_decoding_task(EntropyCodingTaskInfo &task, const type_codeblock &cblk)
{
	switch(cblk.parent_sb->orient)
	{
	case LL:
	case LH:
		task.subband = 0;
		break;
	case HL:
		task.subband = 1;
		break;
	case HH:
		task.subband = 2;
		break;
	}

	task.width = cblk.width;
	task.height = cblk.height;

	task.nominalWidth = cblk.parent_sb->parent_res_lvl->parent_tile_comp->cblk_w;
	task.nominalHeight = cblk.parent_sb->parent_res_lvl->parent_tile_comp->cblk_h;

	task.magbits = cblk.parent_sb->mag_bits;

	task.codeStream = cblk.codestream;
	task.length = cblk.length;
	task.significantBits = cblk.significant_bits;

	//task.coefficients = cblk.data_d;
}

unsigned int reverse(unsigned int in)
{
	unsigned int out = 0;

	for(int i = 0; i < 32; i++)
	{
		out |= ((in >> i) & 1) << (31 - i);
	}

	return out;
}

void decode_tile(type_tile *tile)
{
//	println_start(INFO);

//	start_measure();

	std::list<type_codeblock *> cblks;
	extract_cblks(tile, cblks);

	EntropyCodingTaskInfo *tasks = (EntropyCodingTaskInfo *) malloc(sizeof(EntropyCodingTaskInfo) * cblks.size());

	std::list<type_codeblock *>::iterator ii = cblks.begin();

	int num_tasks = 0;
	for(; ii != cblks.end(); ++ii)
	{
		convert_to_decoding_task(tasks[num_tasks++], *(*ii));
	}

//	printf("%d\n", num_tasks);

	float t = gpuDecode(tasks, num_tasks);

	printf("kernel consumption: %f\n", t);

	ii = cblks.begin();

	for(int i = 0; i < num_tasks; i++, ++ii)
	{
		(*ii)->data_d = tasks[i].coefficients;
	}

	free(tasks);

//	stop_measure(INFO);

//	println_end(INFO);
}
