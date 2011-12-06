/**
 * @file gpu_worker.c
 *
 * @author Milosz Ciznicki
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include "../schedulers/scheduler.h"
#include "worker.h"
#include "worker_helper.h"
#include "../policies/policy.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

int get_ngpus()
{
	int ngpus;

	cudaError_t error;
	error = cudaGetDeviceCount(&ngpus);

	if (error != cudaSuccess)
		printf("Could not get cuda device count!\n");

	return ngpus;
}

static void init_cuda(int32_t device_id)
{
	cudaError_t error;

	error = cudaSetDevice(device_id);
	if (error != cudaSuccess)
		printf("Could not set device id!\n");

	cudaFree(0);
}

static void deinit_cuda()
{
	//	cudaDeviceReset();
}

static void execute_task(hs_worker *worker, _task_t task)
{
	hs_task *_task = task->task;
	task_timing *_task_timing = _task->timing;
	worker_timing *_worker_timing = worker->timing;

	//printf("GPU worker executes task\n");

	if(worker->exectued == 0)
	{
		//printf("GPU zero taks\n");
		get_relative_time(&worker->timing->start_time);
		worker->exectued = 1;
	}

	get_relative_time(&_task_timing->start_time);

	_task->cuda_func(_task->data_interface);

	get_relative_time(&_task_timing->end_time);

	update_worker_exec_status(_worker_timing, &_task_timing->start_time, &_task_timing->end_time);

	fifo_push_task(worker->finished_tasks, task);

	dec_nsubmitted_tasks();
}

void gpu_worker(void *arg)
{
	hs_worker *worker_arg = (hs_worker *) arg;

	bind_to_cpu(worker_arg);

	init_cuda(worker_arg->device_id);

//	printf("GPU I am id:%d\n", worker_arg->worker_id);

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

	deinit_cuda();

	pthread_exit((void*) 0);
}
