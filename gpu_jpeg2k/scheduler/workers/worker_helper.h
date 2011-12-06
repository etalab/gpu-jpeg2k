/**
 * @file worker_helper.h
 *
 * @author Milosz Ciznicki
 */

#ifndef WORKER_HELPER_H_
#define WORKER_HELPER_H_

#include <sched.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

#include "../schedulers/scheduler.h"

static int8_t is_running() {
	return config.running;
}

static void bind_to_cpu(hs_worker *worker_arg)
{
	/* Set affinity mask to bind worker on chosen CPU */
	cpu_set_t cpu_set;
	CPU_ZERO(&cpu_set);
	CPU_SET(worker_arg->worker_id, &cpu_set);

	pthread_t self = pthread_self();

	int ret = pthread_setaffinity_np(self, sizeof(cpu_set_t), &cpu_set);
	if (ret)
	{
		perror("Can not bind thread");
	}
}

static void lock_queue(hs_task_queue *task_queue)
{
	pthread_mutex_lock(&task_queue->lock);
}

static void unlock_queue(hs_task_queue *task_queue)
{
	pthread_mutex_unlock(&task_queue->lock);
}

static void cond_wait_queue(hs_task_queue *task_queue, int8_t arch)
{
//	printf("Wait!\n");
	int ret;
	switch(arch)
	{
	case HS_ARCH_CPU:
		ret = pthread_cond_wait(&task_queue->cpu_activation, &task_queue->lock);
		break;
	case HS_ARCH_CUDA:
		ret = pthread_cond_wait(&task_queue->gpu_activation, &task_queue->lock);
		break;
	}
	if (ret) {
		fprintf(stderr, "pthread_cond_wait : %s", strerror(ret));
		abort();
	}
}

#endif /* WORKER_HELPER_H_ */
