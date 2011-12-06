/*
 * entropy_coding_task.c
 *
 *  Created on: Dec 2, 2011
 *      Author: miloszc
 */

#include "entropy_coding.h"
#include "../tier1/coeff_coder/gpu_coder.h"
#include "../misc/memory_management.cuh"
#include "../print_info/print_info.h"
#include "entropy_coding.h"

void entropy_encoding_cuda_func(void *data_interface)
{
	type_entropy_coding_task *entropy_coding_task = (type_entropy_coding_task *)data_interface;
	EntropyCodingTaskInfo *entropy_tasks = entropy_coding_task->entropy_tasks;
	int num_entropy_tasks = entropy_coding_task->num_entropy_tasks;
	type_subband **sbs = entropy_coding_task->sbs;
	int num_sbs = entropy_coding_task->num_sbs;

//	printf("num entropy: %d\n", num_entropy_tasks);
	long int start_global;
	start_global = start_measure();
/*	int i = 0;
	for(i = 0; i < num_entropy_tasks; ++i) {
		cuda_d_allocate_mem((void **)&(entropy_tasks[i].coefficients), entropy_tasks[i].nominalWidth * entropy_tasks[i].nominalHeight * sizeof(int));
		cuda_memcpy_htd(entropy_tasks[i].h_coefficients, entropy_tasks[i].coefficients, entropy_tasks[i].nominalWidth * entropy_tasks[i].nominalHeight * sizeof(int));
	}*/

	int i = 0, j= 0;
	for(j = 0; j < num_sbs; ++j) {
		type_subband *sb = sbs[j];
		type_tile_comp *tile_comp = sb->parent_res_lvl->parent_tile_comp;
		cuda_d_allocate_mem((void **)&(sb->cblks_data_d), sb->num_cblks * tile_comp->cblk_w * tile_comp->cblk_h * sizeof(int));
		cuda_memcpy_htd(sb->cblks_data_h, sb->cblks_data_d, sb->num_cblks * tile_comp->cblk_w * tile_comp->cblk_h * sizeof(int));
	}

	for(i = 0; i < num_entropy_tasks; ++i) {
        	type_codeblock *cblk = entropy_tasks[i].cblk;
		type_subband *sb = cblk->parent_sb;
		type_tile_comp *tile_comp = sb->parent_res_lvl->parent_tile_comp;
	        entropy_tasks[i].coefficients = sb->cblks_data_d + cblk->cblk_no * tile_comp->cblk_w * tile_comp->cblk_h;
        }
	//printf("copy %ld\n", stop_measure(start_global)/* + copy_time*/);

	long int start_global2;
        start_global2 = start_measure();
	float t = gpuEncode(entropy_tasks, num_entropy_tasks, 0);
	//printf("gpuEncode %ld\n", stop_measure(start_global2)/* + copy_time*/);
	cudaDeviceSynchronize();
}
