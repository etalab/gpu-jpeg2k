/*
 * @file heft_policy.c
 *
 * @author Milosz Ciznicki 
 * @date 19-05-2011
 */

#include "../schedulers/scheduler.h"
#include "policy_helper.h"
#include "policy.h"
#include "../models/model.h"
#include "../queues/fifo_queue.h"
#include "../queues/deque_queue.h"

#include <stdio.h>
#include <float.h>

/* The same queue for all workers */
hs_task_queue *heft_task_queues[MAX_WORKERS];

void init_heft_policy(hs_config *_config, hs_policy *_policy)
{
	int worker;
	for(worker = 0; worker < _config->nworkers; ++worker)
	{
		hs_worker *worker_arg = &_config->workers[worker];

		worker_arg->task_queue = create_fifo_queue();
		heft_task_queues[worker] = worker_arg->task_queue;
	}

	_policy->nqueues = _config->nworkers;

	srand(time(NULL));
}

void deinit_heft_policy(hs_config *_config, hs_policy *_policy)
{
	int worker;
	for(worker = 0; worker < _config->nworkers; ++worker)
	{
		hs_worker *worker_arg = &_config->workers[worker];

		release_fifo(worker_arg->task_queue);
	}
}

void heft_push_task(hs_task_queue *task_queue, _task_t _task)
{
	hs_config *_config = get_config();
	double curr_time_us;
	hs_worker *chosen_worker = NULL;
	double chosen_task_length = 0.0;
	/* chosen actual finish time */
	double chosen_aft = DBL_MAX;

	int worker;
	for(worker = 0; worker < _config->nworkers; ++worker)
	{
		/* current actual finish time */
		double curr_aft;

		hs_worker *worker_arg = &_config->workers[worker];
		hs_deque_queue *curr_queue = worker_arg->task_queue->queue;

		if((_task->task->arch_type & worker_arg->arch) != worker_arg->arch)
		{
			continue;
		}

		curr_time_us = (double)get_relative_time_us();
		curr_queue->est = (curr_queue->est > curr_time_us) ? curr_queue->est : curr_time_us;
		curr_queue->eft = (curr_queue->eft > curr_time_us) ? curr_queue->eft : curr_time_us;

		double task_length = calculate_task_length(worker_arg, _task);

		if(task_length == -1.0)
		{
			chosen_worker = worker_arg;
			chosen_task_length = 0.0;
			break;
		}

		curr_aft = curr_queue->est + curr_queue->length + task_length;

		if(curr_aft < chosen_aft)
		{
			chosen_aft = curr_aft;
			chosen_worker = worker_arg;
			chosen_task_length = task_length;
		}
	}

	if(chosen_worker == NULL)
	{
		printf("Error: No worker was chosen!\n");
	}

	hs_deque_queue *selected_queue = chosen_worker->task_queue->queue;

	selected_queue->eft += chosen_task_length;
	selected_queue->length += chosen_task_length;

	_task->length = chosen_task_length;

	fifo_push_task(chosen_worker->task_queue, _task);
}

_task_t heft_pop_task(hs_task_queue *task_queue)
{
	_task_t _task = fifo_pop_task(task_queue);
	double curr_time_us;

	if(_task)
	{
		hs_deque_queue *queue = task_queue->queue;

		curr_time_us = (double)get_relative_time_us();
		queue->length -= _task->length;
		queue->est = curr_time_us + _task->length;
		queue->eft = queue->est + queue->length;
	}

	return _task;
}

hs_task_queue *heft_get_queue()
{
	return heft_task_queues[0];
}

hs_policy hs_heft_policy = {
		.init = init_heft_policy,
		.deinit = deinit_heft_policy,
		.push_task = heft_push_task,
		.pop_task = heft_pop_task,
		.get_queue = heft_get_queue
};
