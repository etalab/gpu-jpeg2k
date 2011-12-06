/*
 * @file task.c
 *
 * @author Milosz Ciznicki 
 * @date 06-05-2011
 */

#include <pthread.h>
#include <stdio.h>
#include "../policies/policy.h"
#include "task.h"
#include "../timing/timing.h"

static pthread_cond_t submitted_cond = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t submitted_mutex = PTHREAD_MUTEX_INITIALIZER;
static int32_t nsubmitted = 0;

static void init_task(hs_task *task)
{
	task->arch_type = -1;
	task->cpu_func = NULL;
	task->cuda_func = NULL;
	task->data_interface = NULL;
	task->model = NULL;
	task->timing = init_task_timing();
}

hs_task *create_task()
{
	hs_task *task;

	task = (hs_task *) malloc(sizeof(hs_task));

	init_task(task);

	return task;
}

void destroy_task(hs_task *task)
{
	deinit_task_timing(task->timing);
	destroy_model(task->model);
	free(task);
}

void dec_nsubmitted_tasks()
{
	pthread_mutex_lock(&submitted_mutex);

	if(--nsubmitted == 0)
	{
		pthread_cond_broadcast(&submitted_cond);
	}

	pthread_mutex_unlock(&submitted_mutex);
}

void inc_nsubmitted_tasks()
{
	pthread_mutex_lock(&submitted_mutex);

	nsubmitted++;

	pthread_mutex_unlock(&submitted_mutex);
}

void synchronize_tasks()
{
	pthread_mutex_lock(&submitted_mutex);

	while(nsubmitted > 0)
		pthread_cond_wait(&submitted_cond, &submitted_mutex);

	pthread_mutex_unlock(&submitted_mutex);
}

void submit_task(hs_task *task)
{
	get_relative_time(&task->timing->submit_time);

	_task_t _task = _task_new();

	_task->task = task;

	hs_policy *policy = get_policy();

	push_task(policy->get_queue(), _task);

	inc_nsubmitted_tasks();
}
