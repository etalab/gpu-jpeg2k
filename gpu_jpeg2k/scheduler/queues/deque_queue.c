/**
 * @file deque_queue.c
 *
 * @author Milosz Ciznicki
 */

#include <pthread.h>
#include "../schedulers/scheduler.h"
#include "deque_queue.h"
#include "../workers/worker.h"

hs_task_queue *create_deque_queue()
{
	hs_task_queue *task_queue;
	task_queue = (hs_task_queue *)malloc(sizeof(hs_task_queue));

	pthread_mutex_init(&task_queue->lock, NULL);
	pthread_cond_init(&task_queue->cpu_activation, NULL);
	pthread_cond_init(&task_queue->gpu_activation, NULL);

	hs_deque_queue *deque_queue;
	deque_queue = (hs_deque_queue *) malloc(sizeof(hs_deque_queue));

	deque_queue->ntasks = 0;
	deque_queue->ndone = 0;
	deque_queue->queue = _task_list_new();
	deque_queue->est = 0.0;
	deque_queue->eft = 0.0;
	deque_queue->length = 0.0;

	task_queue->queue = deque_queue;

	return task_queue;
}

void release_deque(hs_task_queue *task_queue)
{
	hs_deque_queue *deque_queue = task_queue->queue;

	_task_list_delete(deque_queue->queue);
	free(deque_queue);
	free(task_queue);
}

void deque_push_task_front(hs_task_queue *task_queue, _task_t task)
{
	hs_deque_queue *deque_queue = task_queue->queue;

	pthread_mutex_lock(&task_queue->lock);
	_task_list_push_front(deque_queue->queue, task);
	deque_queue->ntasks++;
	deque_queue->ndone++;
//	printf("Task submitted!\n");
	cond_signal_queue(task_queue, task);
	pthread_mutex_unlock(&task_queue->lock);
}

void deque_push_task_back(hs_task_queue *task_queue, _task_t task)
{
	hs_deque_queue *deque_queue = task_queue->queue;

	pthread_mutex_lock(&task_queue->lock);
	_task_list_push_back(deque_queue->queue, task);
	deque_queue->ntasks++;
	deque_queue->ndone++;
//	printf("Task submitted!\n");
	cond_signal_queue(task_queue, task);
	pthread_mutex_unlock(&task_queue->lock);
}

_task_t deque_pop_task_front(hs_task_queue *task_queue)
{
	hs_deque_queue *deque_queue = task_queue->queue;

	if(deque_queue->ntasks == 0)
	{
		return NULL;
	}

	_task_t task = _task_list_pop_front(deque_queue->queue);

	deque_queue->ntasks--;

	return task;
}

_task_t deque_pop_task_back(hs_task_queue *task_queue)
{
	hs_deque_queue *deque_queue = task_queue->queue;

	if(deque_queue->ntasks == 0)
	{
		return NULL;
	}

	_task_t task = _task_list_pop_back(deque_queue->queue);

	deque_queue->ntasks--;

	return task;
}

int32_t deque_size(hs_task_queue *task_queue)
{
	hs_deque_queue *deque_queue = task_queue->queue;

	return _task_list_size(deque_queue->queue);
}

