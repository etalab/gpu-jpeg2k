/*
 * @file queue.c
 *
 * @author Milosz Ciznicki 
 * @date 06-05-2011
 */

#include <pthread.h>
#include "queue.h"
#include "../policies/policy.h"
#include "../schedulers/scheduler.h"

void all_queues(queue_op op)
{
	hs_policy *policy = get_policy();
	hs_config *config = get_config();
	hs_worker *workers = config->workers;

	int queue;

	for(queue = 0; queue < policy->nqueues; ++queue)
	{
		hs_task_queue *q = workers[queue].task_queue;

		switch(op)
		{
		case LOCK:
			pthread_mutex_lock(&q->lock);
			break;
		case UNLOCK:
			pthread_mutex_unlock(&q->lock);
			break;
		case BROADCAST:
			pthread_cond_broadcast(&q->cpu_activation);
			pthread_cond_broadcast(&q->gpu_activation);
			break;
		}
	}
}

void cond_signal_queue(hs_task_queue *task_queue, _task_t task)
{
	switch(task->task->arch_type)
	{
	case (HS_ARCH_CPU|HS_ARCH_CUDA):
		pthread_cond_signal(&task_queue->cpu_activation);
		pthread_cond_signal(&task_queue->gpu_activation);
		break;
	case HS_ARCH_CPU:
		pthread_cond_signal(&task_queue->cpu_activation);
		break;
	case HS_ARCH_CUDA:
		pthread_cond_signal(&task_queue->gpu_activation);
		break;
	}
}
