/*
 * @file policy.c
 *
 * @author Milosz Ciznicki 
 * @date 05-05-2011
 */

#include "greedy_policy.h"
#include "weighted_rand_policy.h"
#include "work_stealing_policy.h"
#include "heft_policy.h"
#include "../schedulers/scheduler.h"

hs_policy policy;

void init_sched_policy(hs_config *config)
{
	hs_policy *local_policy = &hs_greedy_policy;

	policy.init = local_policy->init;
	policy.deinit = local_policy->deinit;
	policy.get_queue = local_policy->get_queue;
	policy.pop_task = local_policy->pop_task;
	policy.push_task = local_policy->push_task;

	policy.init(config, &policy);
}

_task_t pop_task(hs_task_queue *task_queue)
{
	return policy.pop_task(task_queue);
}

void push_task(hs_task_queue *task_queue, _task_t task)
{
	policy.push_task(task_queue, task);
}

hs_policy *get_policy()
{
	return &policy;
}

void deinit_sched_policy(hs_config *config)
{
	policy.deinit(config, &policy);
}
