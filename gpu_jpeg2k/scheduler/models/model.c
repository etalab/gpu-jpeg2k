/*
 * @file model.c
 *
 * @author Milosz Ciznicki 
 * @date 19-05-2011
 */

#include "../workers/worker.h"
#include "../policies/policy_helper.h"
#include "cost_model.h"

#include <stdio.h>

static double task_length_per_arch(hs_worker *worker_arg, hs_task *task)
{
	hs_model *model = task->model;
	hs_arch_model *arch_model = &model->mutli_arch_model[worker_arg->arch_id];
	double length = -1.0;

	if(arch_model->task_cost)
	{
		length = arch_model->task_cost(task->cost_interface);
	}

	return length;
}

static double task_length_common(hs_worker *worker_arg, hs_task *task)
{
	hs_model *model = task->model;
	hs_arch_model *single_model = &model->single_arch_model;
	double length = -1.0;
	double relative_speed = (double)get_worker_weight_based_on_speed(worker_arg->arch);

	if(single_model->task_cost)
	{
		length = single_model->task_cost(task->cost_interface) / relative_speed;
	}

	return length;
}

double calculate_task_length(hs_worker *worker_arg, _task_t _task)
{
	hs_task *task = _task->task;
	hs_model *model = task->model;

	if(model)
	{
		switch(model->type)
		{
		case HS_ARCH:
			return task_length_per_arch(worker_arg, task);
		case HS_COMMON:
			return task_length_common(worker_arg, task);
		default: printf("No such model was found!\n");
		}
	}

	return 0.0;
}

hs_model *create_model()
{
	hs_model *model = (hs_model *) malloc(sizeof(hs_model));

	return model;
}

void destroy_model(hs_model *model)
{
	if(model)
	{
		free(model);
	}
}

