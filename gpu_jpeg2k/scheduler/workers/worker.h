/**
 * @file worker.h
 *
 * @author Milosz Ciznicki
 */

#ifndef WORKER_H_
#define WORKER_H_

#include "../queues/queue.h"
#include "../timing/timing.h"

#define HS_ARCH_CPU	(1 << 1)
#define HS_ARCH_CUDA	(1 << 2)

typedef struct hs_worker_t {
	hs_task_queue *task_queue;
	pthread_t thread_id;
	pthread_mutex_t mutex;
	pthread_cond_t ready;
	int8_t initialized;
	int8_t exectued;
	int32_t device_id;
	int32_t worker_id;
	int8_t arch;
	int32_t arch_id;
	worker_timing *timing;
	hs_task_queue *finished_tasks;
	char description[32];
} hs_worker;

void init_workers();
void create_workers();
void shutdown_workers();

#endif /* WORKER_H_ */
