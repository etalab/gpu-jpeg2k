/*
 * @file queue.h
 *
 * @author Milosz Ciznicki 
 * @date 05-05-2011
 */

#ifndef QUEUE_H_
#define QUEUE_H_

#include "list.h"
#include "../tasks/task.h"

typedef enum {
	LOCK,
	UNLOCK,
	BROADCAST
} queue_op;

typedef struct hs_task_queue_t {
	void *queue;
	pthread_mutex_t lock;
	pthread_cond_t cpu_activation;
	pthread_cond_t gpu_activation;
} hs_task_queue;

LIST_TYPE(_task, hs_task *task;
double length;)

void all_queues(queue_op op);
void cond_signal_queue(hs_task_queue *task_queue, _task_t task);

#endif /* QUEUE_H_ */
