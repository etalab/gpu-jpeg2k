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
 * mqc_wrapper.cpp
 *
 *  Created on: Dec 9, 2011
 *      Author: miloszc
 */

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include "mqc_wrapper.h"
#include "mqc_common.h"
#include "../gpu_coeff_coder2.cuh"
#include "mqc_develop.h"
extern "C" {
#include "timer.h"
#include "mqc_data.h"
}

static void mqc_gpu_encode_(EntropyCodingTaskInfo *infos, CodeBlockAdditionalInfo* mqc_data, int codeBlocks,
		unsigned char* d_cxds, int maxOutLength, const char* param = 0, const char* name_suffix = 0) {
	// Initialize CUDA
	cudaError cuerr = cudaSuccess;

	// Determine cxd blocks count
	int cxd_block_count = codeBlocks;
	// Allocate CPU memory for cxd blocks
	struct cxd_block* cxd_blocks = new struct cxd_block[cxd_block_count];
	// Fill CPU memory with cxd blocks
	int cxd_index = 0;
	int byte_index = 0;
	int cxd_block_index = 0;
	for (int cblk_index = 0; cblk_index < codeBlocks; cblk_index++) {
		cxd_blocks[cxd_block_index].cxd_begin = cxd_index;
		cxd_blocks[cxd_block_index].cxd_count = mqc_data[cblk_index].length > 0 ? mqc_data[cblk_index].length : 0;
		cxd_blocks[cxd_block_index].byte_begin = byte_index;
		cxd_blocks[cxd_block_index].byte_count = cxd_get_buffer_size(cxd_blocks[cxd_block_index].cxd_count);

		cxd_index += /*cxd_blocks[cxd_block_index].cxd_count*/maxOutLength;
		byte_index += cxd_blocks[cxd_block_index].byte_count;
		cxd_block_index++;
	}

	// Allocate GPU memory for cxd blocks
	struct cxd_block* d_cxd_blocks;
	cuerr = cudaMalloc((void**) &d_cxd_blocks, cxd_block_count * sizeof(struct cxd_block));
	if (cuerr != cudaSuccess) {
		std::cerr << "Can't allocate device memory for cxd blocks: " << cudaGetErrorString(cuerr) << std::endl;
		return;
	}
	// Fill GPU memory with cxd blocks
	cudaMemcpy((void*) d_cxd_blocks, (void*) cxd_blocks, cxd_block_count * sizeof(struct cxd_block),
			cudaMemcpyHostToDevice);

	// Allocate GPU memory for output bytes
	unsigned char* d_bytes;
	cuerr = cudaMalloc((void**) &d_bytes, 1 + byte_index * sizeof(unsigned char));
	if (cuerr != cudaSuccess) {
		std::cerr << "Can't allocate device memory for output bytes: " << cudaGetErrorString(cuerr) << std::endl;
		return;
	}

	// Zero memory and move pointer by one (encoder access bp-1 so we must have that position)
	cuerr = cudaMemset((void*) d_bytes, 0, 1 + byte_index * sizeof(unsigned char));
	if (cuerr != cudaSuccess) {
		std::cout << "Can't memset for output bytes: " << cudaGetErrorString(cuerr) << std::endl;
		return;
	}
	d_bytes++;

	// Init encoder
//    mqc_gpu_init(param);
	mqc_gpu_develop_init(param);

	// Make runs
	double * duration = new double[1];
	bool correct = true;
	for (int run_index = 0; run_index < 1; run_index++) {

		struct timer_state timer_state;
		timer_reset(&timer_state);
		timer_start(&timer_state);

		// Encode on GPU
//        mqc_gpu_encode(d_cxd_blocks,cxd_block_count,d_cxds,d_bytes);
		mqc_gpu_develop_encode(d_cxd_blocks, cxd_block_count, d_cxds, d_bytes);

		duration[run_index] = timer_stop(&timer_state);

		printf("mqc %f\n", duration[run_index]);

		// TODO: Check correction for carry bit
		// It seems like it is not needed but who nows, check it for sure

		// Allocate CPU memory for output bytes
		unsigned char* bytes = new unsigned char[byte_index];
		// Copy output bytes to CPU memory
		cuerr = cudaMemcpy((void*) bytes, (void*) d_bytes, byte_index * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		if (cuerr != cudaSuccess) {
			std::cout << "Can't memcpy for output bytes: " << cudaGetErrorString(cuerr) << std::endl;
			return;
		}
		// Copy cxd blocks to CPU memory
		cuerr = cudaMemcpy((void*) cxd_blocks, (void*) d_cxd_blocks, cxd_block_count * sizeof(struct cxd_block),
				cudaMemcpyDeviceToHost);
		if (cuerr != cudaSuccess) {
			std::cout << "Can't memcpy for output d_cxd_blocks: " << cudaGetErrorString(cuerr) << std::endl;
			return;
		}

		// Check output bytes
		int cblk_index;
		for (cblk_index = 0; cblk_index < codeBlocks; cblk_index++) {
			EntropyCodingTaskInfo * cblk = &infos[cblk_index];
			struct cxd_block* cxd_block = &cxd_blocks[cblk_index];
			cblk->length = cxd_block->byte_count > 0 ? cxd_block->byte_count : 0;
			cblk->codeStream = (unsigned char *)my_malloc(sizeof(unsigned char) * cxd_block->byte_count);
			memcpy(cblk->codeStream, &bytes[cxd_block->byte_begin], sizeof(unsigned char) * cxd_block->byte_count);
		}

		// Free CPU Memory
//		delete[] bytes;
	}

	// Deinit encoder
	//    mqc_gpu_deinit();
	mqc_gpu_develop_deinit();

	// Free GPU memory
	cudaFree((void*) --d_bytes);
	cudaFree((void*) d_cxd_blocks);

	// Free CPU memory
	delete[] cxd_blocks;
}

void mqc_gpu_encode_test() {
	char file_name[128];
	sprintf(file_name, "/home/miloszc/Projects/images/rgb8bit/flower_foveon.ppm\0");
	struct mqc_data* mqc_data = mqc_data_create_from_image(file_name);
	if(mqc_data == 0) {
		std::cerr << "Can't receive data from openjpeg: " << std::endl;
		return;
	}

	int maxOutLength = (4096 * 10);

    // Initialize CUDA
    cudaError cuerr = cudaSuccess;

    // Determine cxd count
    int cxd_count = 0;
    int byte_index = 0;
    for ( int cblk_index = 0; cblk_index < mqc_data->cblk_count; cblk_index++ ) {
    	struct mqc_data_cblk *cblk = mqc_data->cblks[cblk_index];
    	cxd_count += cblk->cxd_count;
    	byte_index += cblk->byte_count;
    }

    // Allocate CPU memory for CX,D pairs
//    int cxd_size = cxd_array_size(cxd_count);
    int cxd_size = mqc_data->cblk_count * maxOutLength;
    unsigned char* cxds = new unsigned char[cxd_size];
    memset(cxds,0,cxd_size);
    // Fill CPU memory with CX,D pairs

    for ( int cblk_index = 0; cblk_index < mqc_data->cblk_count; cblk_index++ ) {
        struct mqc_data_cblk* cblk = mqc_data->cblks[cblk_index];
        int index = cblk_index * maxOutLength;
        for ( int cxd_index = 0; cxd_index < cblk->cxd_count; cxd_index++ ) {
            cxd_array_put(cxds, index, cblk->cxds[cxd_index].cx, cblk->cxds[cxd_index].d);
            index++;
        }
    }

    // Allocate GPU memory for CX,D pairs
    unsigned char* d_cxds;
    cuerr = cudaMalloc((void**)&d_cxds, cxd_size * sizeof(unsigned char));
    if ( cuerr != cudaSuccess ) {
        std::cerr << "Can't allocate device memory for cxd pairs: " << cudaGetErrorString(cuerr) << std::endl;
        return;
    }
    // Fill GPU memory with CX,D pairs
    cudaMemcpy((void*)d_cxds, (void*)cxds, cxd_size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int codeBlocks = mqc_data->cblk_count;
    CodeBlockAdditionalInfo *h_infos = (CodeBlockAdditionalInfo *) my_malloc(sizeof(CodeBlockAdditionalInfo) * codeBlocks);

    for(int i = 0; i < codeBlocks; ++i) {
    	struct mqc_data_cblk* cblk = mqc_data->cblks[i];
    	h_infos[i].length = cblk->cxd_count;
    }

    EntropyCodingTaskInfo *infos = (EntropyCodingTaskInfo *) my_malloc(sizeof(EntropyCodingTaskInfo) * codeBlocks);

	mqc_gpu_encode_(infos, h_infos, codeBlocks, d_cxds, maxOutLength);

	// Check output bytes
	int cblk_index;
	int byte_count = 0;
	bool correct = true;
	for ( cblk_index = 0; cblk_index < mqc_data->cblk_count; cblk_index++ ) {
		struct mqc_data_cblk* cblk = mqc_data->cblks[cblk_index];
		EntropyCodingTaskInfo* cxd_block = &infos[cblk_index];
		byte_count += cxd_block->length;
		if ( cblk->byte_count != cxd_block->length ) {
			correct = false;
			std::cerr << "WRONG at block [" << cblk_index << "], because byte count [";
			std::cerr << cxd_block->length << "] != [" << cblk->byte_count << "]";
			break;
		} else {
			for ( int byte_index = 0; byte_index < cblk->byte_count; byte_index++ ) {
				if ( cxd_block->codeStream[byte_index] != cblk->bytes[byte_index] ) {
					correct = false;
					std::cerr << "WRONG at block [" << cblk_index << "] at byte [" << byte_index << "]";
					std::cerr << " because [" << (int)cxd_block->codeStream[byte_index];
					std::cerr << "] != [" << (int)cblk->bytes[byte_index] << "]";
					break;
				}
			}
			if ( correct == false )
				break;
		}
	}

	delete[] cxds;
	cudaFree((void *)d_cxds);
}

void mqc_gpu_encode(EntropyCodingTaskInfo *infos, CodeBlockAdditionalInfo* h_infos, int codeBlocks,
		unsigned char *d_cxd_pairs, int maxOutLength) {
// Initialize CUDA
	cudaError cuerr = cudaSuccess;

/*	int cxd_count = 0;

	for (int i = 0; i < codeBlocks; ++i) {
		cxd_count += h_infos[i].length > 0 ? h_infos[i].length : 0;
//		if(i < 1000)
//			printf("%d) %d\n", i, h_infos[i].length);
	}

// Allocate GPU memory for CX,D pairs
	unsigned char* d_cxds;
	cuerr = cudaMalloc((void**) &d_cxds, cxd_count * sizeof(unsigned char));
	if (cuerr != cudaSuccess) {
		std::cerr << "Can't allocate device memory for cxd pairs: " << cudaGetErrorString(cuerr) << std::endl;
		return;
	}

	int cxd_idx = 0;
	for (int i = 0; i < codeBlocks; i++) {
		if (h_infos[i].length > 0) {
			int len = h_infos[i].length;
//			printf("%d) %d\n", i, len);
			cuerr = cudaMemcpy((void*) (d_cxds + cxd_idx), (void*) (d_cxd_pairs + i * maxOutLength), len * sizeof(unsigned char),
					cudaMemcpyDeviceToDevice);
			if (cuerr != cudaSuccess) {
				std::cerr << "Can't copy device memory for input CX,D pairs: " << cudaGetErrorString(cuerr)
						<< std::endl;
				return;
			}
			cxd_idx += h_infos[i].length;
		}
	}*/

	mqc_gpu_encode_(infos, h_infos, codeBlocks, /*d_cxds*/d_cxd_pairs, maxOutLength);
//	mqc_gpu_encode_test();

//	cudaFree((void *)d_cxds);
}
