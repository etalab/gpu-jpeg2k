/*
 * entropy_coding.c
 *
 *  Created on: Dec 2, 2011
 *      Author: miloszc
 */

#ifndef ENTROPY_CODING_C_
#define ENTROPY_CODING_C_

#include "../tier1/coeff_coder/gpu_coder.h"
#include "../types/image_types.h"

typedef struct
{
	EntropyCodingTaskInfo *entropy_tasks;
	int num_entropy_tasks;
	type_subband **sbs;
	int num_sbs;

} type_entropy_coding_task;

void encode_tasks_multigpu(type_tile *tile, EntropyCodingTaskInfo *tasks, int num_tasks, type_coding_param *coding_params);

#endif /* ENTROPY_CODING_C_ */
