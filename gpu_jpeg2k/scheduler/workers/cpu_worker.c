/**
 * @file cpu_worker.c
 *
 * @author Milosz Ciznicki
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include "../schedulers/scheduler.h"
#include "worker.h"
#include "worker_helper.h"
#include "../policies/policy.h"
#include "../timing/timing.h"
#include "common_worker.h"

int get_ncpus()
{
	return sysconf(_SC_NPROCESSORS_ONLN);
}

static void execute_task(hs_worker *worker, _task_t task)
{
	hs_task *_task = task->task;
	task_timing *_task_timing = _task->timing;
	worker_timing *_worker_timing = worker->timing;

	//printf("%c[%d;%dmCPU worker executes task%c[%dm\n", 27, 1, 33, 27, 0);

	if(worker->exectued == 0)
	{
		//printf("%c[%d;%dmCPU zero tasks %c[%dm\n", 27, 1, 33, 27, 0);
		get_relative_time(&worker->timing->start_time);
		worker->exectued = 1;
	}

	get_relative_time(&_task_timing->start_time);

	_task->cpu_func(_task->data_interface);

	get_relative_time(&_task_timing->end_time);

	update_worker_exec_status(_worker_timing, &_task_timing->start_time, &_task_timing->end_time);

	fifo_push_task(worker->finished_tasks, task);

	dec_nsubmitted_tasks();
}

void cpu_worker(void *arg)
{
	hs_worker *worker_arg = (hs_worker *) arg;

	bind_to_cpu(worker_arg);

	printf("CPU I am id:%d\n", worker_arg->worker_id);

	pthread_mutex_lock(&worker_arg->mutex);
	worker_arg->initialized = 1;
	pthread_cond_signal(&worker_arg->ready);
	pthread_mutex_unlock(&worker_arg->mutex);

	_task_t task;
	while (is_running())
	{
		lock_queue(worker_arg->task_queue);

		task = pop_task(worker_arg->task_queue);

		if (task == NULL)
		{
			if (is_running())
				sleep_worker(worker_arg);

			unlock_queue(worker_arg->task_queue);
			continue;
		}

		unlock_queue(worker_arg->task_queue);

		if ((task->task->arch_type & worker_arg->arch) != worker_arg->arch)
		{
			push_task(worker_arg->task_queue, task);
			continue;
		}

		execute_task(worker_arg, task);
	}

	pthread_exit(NULL);
}
