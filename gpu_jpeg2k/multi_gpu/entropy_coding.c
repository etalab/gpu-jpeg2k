/*
 * entropy_coding.c
 *
 *  Created on: Dec 2, 2011
 *      Author: miloszc
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "entropy_coding.h"
#include "../misc/memory_management.cuh"
#include "../misc/cuda_errors.h"
#include "../print_info/print_info.h"
#include "../scheduler/schedulers/scheduler.h"
#include "entropy_coding_task.h"

void copy_sb_coeff_to_host(type_tile *tile, EntropyCodingTaskInfo *tasks)
{
	int counter = 0;
	int i = 0, j = 0, k = 0, l = 0;
	for(i = 0; i < tile->parent_img->num_components; i++)
	{
		type_tile_comp *tile_comp = &(tile->tile_comp[i]);
		for(j = 0; j < tile_comp->num_rlvls; j++)
		{
			type_res_lvl *res_lvl = &(tile_comp->res_lvls[j]);
			for(k = 0; k < res_lvl->num_subbands; k++)
			{
				type_subband *sb = &(res_lvl->subbands[k]);
				cuda_h_allocate_mem((void **) &(sb->cblks_data_h), sb->num_cblks * tile_comp->cblk_w * tile_comp->cblk_h * sizeof(int));
				cuda_memcpy_dth(sb->cblks_data_d, sb->cblks_data_h, sb->num_cblks * tile_comp->cblk_w * tile_comp->cblk_h * sizeof(int));
				for(l = 0; l < sb->num_cblks; l++)
				{
					type_codeblock *cblk = &(sb->cblks[l]);
					tasks[counter].h_coefficients = sb->cblks_data_h + cblk->cblk_no * tile_comp->cblk_w * tile_comp->cblk_h;
					tasks[counter++].cblk = cblk;
				}
				cuda_d_free(sb->cblks_data_d);
			}
		}
	}
}

void copy_coeff_to_host(EntropyCodingTaskInfo *task) {
	cuda_h_allocate_mem((void **)&(task->h_coefficients), task->nominalWidth * task->nominalHeight * sizeof(int));
	cuda_memcpy_dth(task->coefficients, task->h_coefficients, task->nominalWidth * task->nominalHeight * sizeof(int));
	task->coefficients = NULL;
}

void copy_task(EntropyCodingTaskInfo *src, EntropyCodingTaskInfo *dst) {
	memcpy(dst, src, sizeof(EntropyCodingTaskInfo));
	dst->h_coefficients = src->h_coefficients;
	dst->coefficients = src->coefficients;
	dst->cblk = src->cblk;
}

void init_task_data(hs_task *task, type_entropy_coding_task *entropy_coding_tasks_package) {
	task->data_interface = (void *)entropy_coding_tasks_package;
}

void gather_task_data(EntropyCodingTaskInfo *entropy_task_package, int num_tasks_in_package) {

}

void compute_tasks(EntropyCodingTaskInfo *entropy_tasks, int num_entropy_tasks, type_coding_param *coding_params) {
	int npackages = 2;
	if(npackages > num_entropy_tasks) {
		fprintf(stderr, "Error occurred npackages < num_entropy_tasks.\n");
		exit(1);
	}

	hs_task **tasks = (hs_task **) malloc(npackages * sizeof(hs_task *));

	type_entropy_coding_task *entropy_coding_tasks_packages = (type_entropy_coding_task *) malloc(npackages * sizeof(type_entropy_coding_task));
//	EntropyCodingTaskInfo **entropy_task_packages = (EntropyCodingTaskInfo **) malloc(npackages * sizeof(EntropyCodingTaskInfo *));

//	int *tasks_per_package = (int *) malloc(npackages * sizeof(int));
	int nominal_package_size = num_entropy_tasks / npackages;

	int k = 0;
	for(k = 0; k < npackages; ++k) {
//		tasks_per_package[k] = nominal_package_size;
		entropy_coding_tasks_packages[k].num_entropy_tasks = nominal_package_size;
	}
//	tasks_per_package[npackages - 1] += num_entropy_tasks % nominal_package_size;
	entropy_coding_tasks_packages[npackages - 1].num_entropy_tasks += num_entropy_tasks % nominal_package_size;

	int i = 0;
	int curr_sbs = 0;
	for(k = 0; k < npackages; ++k) {
//		entropy_task_packages[k] = (EntropyCodingTaskInfo *) malloc(tasks_per_package[k] * sizeof(EntropyCodingTaskInfo));
		curr_sbs = 0;
		entropy_coding_tasks_packages[k].entropy_tasks = (EntropyCodingTaskInfo *) malloc(entropy_coding_tasks_packages[k].num_entropy_tasks * sizeof(EntropyCodingTaskInfo));
		entropy_coding_tasks_packages[k].sbs = (type_subband **) malloc(100 * sizeof(type_subband *));
		copy_task(&entropy_tasks[k * nominal_package_size], &entropy_coding_tasks_packages[k].entropy_tasks[0]);
		type_subband *sb = entropy_coding_tasks_packages[k].entropy_tasks[0].cblk->parent_sb;
//		type_subband *sb/* = entropy_tasks[k * nominal_package_size].cblk->parent_sb*/;
		entropy_coding_tasks_packages[k].sbs[curr_sbs++] = sb;
		for(i = 1; i < entropy_coding_tasks_packages[k].num_entropy_tasks; ++i) {
			copy_task(&entropy_tasks[k * nominal_package_size + i], &entropy_coding_tasks_packages[k].entropy_tasks[i]);
			sb = entropy_coding_tasks_packages[k].entropy_tasks[i].cblk->parent_sb;
//			copy_task(&entropy_tasks[k * nominal_package_size + i], &entropy_task_packages[k][i]);
			if(sb != entropy_coding_tasks_packages[k].sbs[curr_sbs - 1]) {
				entropy_coding_tasks_packages[k].sbs[curr_sbs++] = sb;
			}
		}
		entropy_coding_tasks_packages[k].num_sbs = curr_sbs;
	}

	for(i = 0; i < npackages; ++i)
	{
		tasks[i] = create_task();

		init_task_data(tasks[i], &entropy_coding_tasks_packages[i]);

		tasks[i]->arch_type = HS_ARCH_CUDA;
		tasks[i]->cuda_func = entropy_encoding_cuda_func;

		submit_task(tasks[i]);
	}

	synchronize_tasks();

//	get_timing_results_from_exec_time();

	for(k = 0; k < npackages; ++k) {
		for(i = 0; i < entropy_coding_tasks_packages[k].num_entropy_tasks; ++i) {
			copy_task(&entropy_coding_tasks_packages[k].entropy_tasks[i], &entropy_tasks[k * nominal_package_size + i]);
		}
	}

	free(tasks);

//	pthread_exit(NULL);
}

void encode_tasks_multigpu(type_tile *tile, EntropyCodingTaskInfo *tasks, int num_tasks, type_coding_param *coding_params) {
//	printf("num tasks:%d", num_tasks);
	long int start_global;
	start_global = start_measure();
/*	int k = 0;
	for(k = 0; k < num_tasks; ++k)
	{
		copy_coeff_to_host(&(tasks[k]));
	}*/
	copy_sb_coeff_to_host(tile, tasks);
	cudaThreadSynchronize();
	//printf("1 gl %ld\n", stop_measure(start_global));
	long int start_global2;
        start_global2 = start_measure();
	compute_tasks(tasks, num_tasks, coding_params);
	cudaThreadSynchronize();
        //printf("2 gl %ld\n", stop_measure(start_global2));
}
