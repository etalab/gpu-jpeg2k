/*
 * @file greedy_policy.c
 *
 * @author Milosz Ciznicki 
 * @date 05-05-2011
 */

#include "policy.h"
#include "../queues/fifo_queue.h"

/* The same queue for all workers */
hs_task_queue *greedy_task_queue;

void init_greedy_policy(hs_config *_config, hs_policy *_policy)
{
	greedy_task_queue = create_fifo_queue();

	int worker;
	for(worker = 0; worker < _config->nworkers; ++worker)
	{
		hs_worker *worker_arg = &_config->workers[worker];

		worker_arg->task_queue = greedy_task_queue;
	}

	_policy->nqueues = 1;
}

void deinit_greedy_policy(hs_config *_config, hs_policy *_policy)
{
	release_fifo(greedy_task_queue);
}

void greedy_push_task(hs_task_queue *task_queue, _task_t task)
{
	fifo_push_task(task_queue, task);
}

_task_t greedy_pop_task(hs_task_queue *task_queue)
{
	return fifo_pop_task(task_queue);
}

hs_task_queue *greedy_get_queue()
{
	return greedy_task_queue;
}

hs_policy hs_greedy_policy = {
		.init = init_greedy_policy,
		.deinit = deinit_greedy_policy,
		.push_task = greedy_push_task,
		.pop_task = greedy_pop_task,
		.get_queue = greedy_get_queue
};
