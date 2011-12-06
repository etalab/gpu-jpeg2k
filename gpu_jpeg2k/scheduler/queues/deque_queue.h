/**
 * @file deque_queue.h
 *
 * @author Milosz Ciznicki
 */

#ifndef DEQUE_QUEUE_H_
#define DEQUE_QUEUE_H_

#include "queue.h"

typedef struct hs_deque_queue_t {
	_task_list_t queue;

	int32_t ntasks;
	int32_t ncpu_tasks;
	int32_t ngpu_tasks;
	int32_t ndone;

	/* earliest execution start time */
	double est;
	/* earliest execution finish time */
	double eft;
	/* scheduled tasks length */
	double length;
} hs_deque_queue;

hs_task_queue *create_deque_queue();
void release_deque(hs_task_queue *task_queue);
void deque_push_task_front(hs_task_queue *task_queue, _task_t task);
void deque_push_task_back(hs_task_queue *task_queue, _task_t task);
_task_t deque_pop_task_front(hs_task_queue *task_queue);
_task_t deque_pop_task_back(hs_task_queue *task_queue);
int32_t deque_size(hs_task_queue *task_queue);

#endif /* DEQUE_QUEUE_H_ */
