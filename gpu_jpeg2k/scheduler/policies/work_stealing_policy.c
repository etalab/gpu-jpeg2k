/**
 * @file work_stealing_policy.c
 *
 * @author Milosz Ciznicki
 */

#include "../schedulers/scheduler.h"
#include "policy_helper.h"
#include "policy.h"
#include "../queues/deque_queue.h"

#include <stdio.h>

static int curr_worker;

void init_work_stealing_policy(hs_config *_config, hs_policy *_policy)
{
	int worker;
	for(worker = 0; worker < _config->nworkers; ++worker)
	{
		hs_worker *worker_arg = &_config->workers[worker];

		worker_arg->task_queue = create_deque_queue();
	}

	_policy->nqueues = _config->nworkers;
	curr_worker = 0;
}

void deinit_work_stealing_policy(hs_config *_config, hs_policy *_policy)
{
	int worker;
	for(worker = 0; worker < _config->nworkers; ++worker)
	{
		hs_worker *worker_arg = &_config->workers[worker];

		release_deque(worker_arg->task_queue);
	}
}

void work_stealing_push_task(hs_task_queue *task_queue, _task_t task)
{
	deque_push_task_front(task_queue, task);
}

static hs_task_queue *select_victim()
{
	hs_config *_config = get_config();
	hs_worker *worker_arg = &_config->workers[curr_worker];

	printf("worker_id %d\n", worker_arg->worker_id);
	curr_worker = (curr_worker + 1) % _config->nworkers;

	return worker_arg->task_queue;
}

_task_t work_stealing_pop_task(hs_task_queue *task_queue)
{
	_task_t task = deque_pop_task_front(task_queue);

	if(task)
	{
		return task;
	}

//	hs_task_queue *victim_queue = select_victim();

	hs_config *_config = get_config();
	int worker;
	for(worker = 0; worker < _config->nworkers; ++worker)
	{
//		hs_worker *worker_arg = &_config->workers[worker];
//		hs_task_queue *victim_queue = worker_arg->task_queue;
		hs_task_queue *victim_queue = select_victim();

		if(!pthread_mutex_trylock(&victim_queue->lock))
		{
//			printf("steal\n");
			task = deque_pop_task_back(victim_queue);
			pthread_mutex_unlock(&victim_queue->lock);
			if(task)
			{
				break;
			}
		}
	}

	return task;
}

static hs_task_queue *select_queue()
{
	return select_victim();
}

hs_task_queue *work_stealing_get_queue()
{
	return select_queue();
}

hs_policy hs_work_stealing_policy = {
		.init = init_work_stealing_policy,
		.deinit = deinit_work_stealing_policy,
		.push_task = work_stealing_push_task,
		.pop_task = work_stealing_pop_task,
		.get_queue = work_stealing_get_queue
};
