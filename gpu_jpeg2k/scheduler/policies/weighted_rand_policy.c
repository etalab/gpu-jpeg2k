/**
 * @file wieghted_rand_policy.c
 *
 * @author Milosz Ciznicki
 */

#include "../schedulers/scheduler.h"
#include "policy_helper.h"
#include "policy.h"
#include "../queues/fifo_queue.h"

/* The same queue for all workers */
hs_task_queue *weighted_rand_task_queues[MAX_WORKERS];

void init_weighted_rand_policy(hs_config *_config, hs_policy *_policy)
{
	int worker;
	for(worker = 0; worker < _config->nworkers; ++worker)
	{
		hs_worker *worker_arg = &_config->workers[worker];

		worker_arg->task_queue = create_fifo_queue();
		weighted_rand_task_queues[worker] = worker_arg->task_queue;
	}

	_policy->nqueues = _config->nworkers;

	srand(time(NULL));
}

void deinit_weighted_rand_policy(hs_config *_config, hs_policy *_policy)
{
	int worker;
	for(worker = 0; worker < _config->nworkers; ++worker)
	{
		hs_worker *worker_arg = &_config->workers[worker];

		release_fifo(worker_arg->task_queue);
	}
}

void weighted_rand_push_task(hs_task_queue *task_queue, _task_t task)
{
	float weight_sum = 0.0f;
	hs_config *_config = get_config();

	int worker;
	for(worker = 0; worker < _config->nworkers; ++worker)
	{
		hs_worker *worker_arg = &_config->workers[worker];

		weight_sum += get_worker_weight_based_on_speed(worker_arg->arch);
	}

	float r = ((float)rand()/(float)RAND_MAX) * weight_sum;
//	printf("r=%f\n", r);
	float curr_wiehgt = 0.0f;
	hs_task_queue *selected_queue = NULL;

	for(worker = 0; worker < _config->nworkers; ++worker)
	{
		hs_worker *worker_arg = &_config->workers[worker];
		float worker_weight = get_worker_weight_based_on_speed(worker_arg->arch);

		curr_wiehgt += worker_weight;

		if(curr_wiehgt > r)
		{
			selected_queue = worker_arg->task_queue;
			break;
		}
	}

	fifo_push_task(selected_queue, task);
}

_task_t weighted_rand_pop_task(hs_task_queue *task_queue)
{
	return fifo_pop_task(task_queue);
}

hs_task_queue *weighted_rand_get_queue()
{
	return weighted_rand_task_queues[0];
}

hs_policy hs_weighted_rand_policy = {
		.init = init_weighted_rand_policy,
		.deinit = deinit_weighted_rand_policy,
		.push_task = weighted_rand_push_task,
		.pop_task = weighted_rand_pop_task,
		.get_queue = weighted_rand_get_queue
};
