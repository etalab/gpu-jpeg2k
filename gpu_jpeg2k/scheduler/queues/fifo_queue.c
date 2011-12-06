/*
 * @file fifo_queue.c
 *
 * @author Milosz Ciznicki 
 * @date 05-05-2011
 */

#include <pthread.h>
#include "../schedulers/scheduler.h"
#include "fifo_queue.h"
#include "deque_queue.h"
#include "../workers/worker.h"

hs_task_queue *create_fifo_queue()
{
	return create_deque_queue();
}

void release_fifo(hs_task_queue *task_queue)
{
	release_deque(task_queue);
}

void fifo_push_task(hs_task_queue *task_queue, _task_t task)
{
	deque_push_task_front(task_queue, task);
}

_task_t fifo_pop_task(hs_task_queue *task_queue)
{
	return deque_pop_task_back(task_queue);
}

int32_t fifo_size(hs_task_queue *task_queue)
{
	return deque_size(task_queue);
}
