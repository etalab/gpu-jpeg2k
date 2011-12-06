/*
 * @file fifo_queue.h
 *
 * @author Milosz Ciznicki 
 * @date 05-05-2011
 */

#ifndef FIFO_QUEUE_H_
#define FIFO_QUEUE_H_

#include "queue.h"

/*typedef struct hs_fifo_queue_t {
	_task_list_t queue;

	int32_t ntasks;
	int32_t ncpu_tasks;
	int32_t ngpu_tasks;
	int32_t ndone;
} hs_fifo_queue;*/

hs_task_queue *create_fifo_queue();
void release_fifo(hs_task_queue *task_queue);
void fifo_push_task(hs_task_queue *task_queue, _task_t task);
_task_t fifo_pop_task(hs_task_queue *task_queue);
int32_t fifo_size(hs_task_queue *task_queue);

#endif /* FIFO_QUEUE_H_ */
