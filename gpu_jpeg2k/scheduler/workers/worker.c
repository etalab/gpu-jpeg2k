/**
 * @file worker.c
 *
 * @author Milosz Ciznicki
 */

#include <stdlib.h>
#include <stdio.h>

#include "worker.h"
#include "../schedulers/scheduler.h"
#include "../queues/fifo_queue.h"
#include "cpu_worker.h"
#include "gpu_worker.h"
#include "../timing/timing.h"

void init_workers()
{
	config.nhcpus = 0/*get_ncpus()*/;
	config.nhgpus = get_ngpus();
	config.nworkers = 0;

	/* Initialize gpu workers */
	int gpu;
	config.ngpus = config.nhgpus;
	for(gpu = 0; gpu < config.ngpus; ++gpu)
	{
		hs_worker *worker_arg = &config.workers[config.nworkers + gpu];
		worker_arg->worker_id = config.nworkers + gpu;
		worker_arg->arch_id = HS_CUDA_ID;
		worker_arg->task_queue = NULL;
		worker_arg->device_id = gpu;
		worker_arg->initialized = 0;
		worker_arg->exectued = 0;
		worker_arg->arch = HS_ARCH_CUDA;
		worker_arg->timing = init_worker_timing();
		worker_arg->finished_tasks = create_fifo_queue();
		pthread_cond_init(&worker_arg->ready, NULL);
		pthread_mutex_init(&worker_arg->mutex, NULL);
		sprintf(worker_arg->description, "GPU:%d\0", worker_arg->worker_id);
	}
	config.nworkers += config.ngpus;

	/* Initialize cpu workers */
	int cpu;
	if(config.nhcpus >= config.ngpus)
		config.ncpus = config.nhcpus - config.ngpus;
	else
		config.ncpus = 0;
	for(cpu = 0; cpu < config.ncpus; ++cpu)
	{
		hs_worker *worker_arg = &config.workers[config.nworkers + cpu];
		worker_arg->worker_id = config.nworkers + cpu;
		worker_arg->arch_id = HS_CPU_ID;
		worker_arg->task_queue = NULL;
		worker_arg->device_id = cpu;
		worker_arg->initialized = 0;
		worker_arg->arch = HS_ARCH_CPU;
		worker_arg->timing = init_worker_timing();
		worker_arg->finished_tasks = create_fifo_queue();
		pthread_cond_init(&worker_arg->ready, NULL);
		pthread_mutex_init(&worker_arg->mutex, NULL);
		sprintf(worker_arg->description, "CPU:%d\0", worker_arg->worker_id);
	}
	config.nworkers += config.ncpus;

	printf("cpus:%d gpus:%d\n", config.ncpus, config.ngpus);
}

void create_workers()
{
	config.running = 1;

	/* Create workers */
	int worker;
	for(worker = 0; worker < config.nworkers; worker++)
	{
		hs_worker *worker_arg = &config.workers[worker];
		switch(worker_arg->arch)
		{
		case HS_ARCH_CPU:
			pthread_create(&worker_arg->thread_id, NULL, cpu_worker, (void *) worker_arg);
			break;
		case HS_ARCH_CUDA:
			pthread_create(&worker_arg->thread_id, NULL, gpu_worker, (void *) worker_arg);
			break;
		}
	}

	/* Wait until all workers are initialized */
	for(worker = 0; worker < config.nworkers; worker++)
	{
		hs_worker *worker_arg = &config.workers[worker];
		pthread_mutex_lock(&worker_arg->mutex);
		while (!worker_arg->initialized)
			pthread_cond_wait(&worker_arg->ready, &worker_arg->mutex);
		pthread_mutex_unlock(&worker_arg->mutex);
	}
}

void shutdown_workers()
{
	all_queues(LOCK);

	config.running = 0;

	all_queues(BROADCAST);

	all_queues(UNLOCK);

	int worker;
	for(worker = 0; worker < config.nworkers; worker++)
	{
		hs_worker *worker_arg = &config.workers[worker];

		if (!pthread_equal(pthread_self(), worker_arg->thread_id))
		{
			pthread_join(worker_arg->thread_id, NULL);
		}
	}
}

void deinit_workers()
{
	/* Destory workers */
	int worker;
	for(worker = 0; worker < config.nworkers; worker++)
	{
		hs_worker *worker_arg = &config.workers[worker];

		deinit_worker_timing(worker_arg->timing);

		_task_t _task = fifo_pop_task(worker_arg->finished_tasks);
		for(;_task != NULL;_task = fifo_pop_task(worker_arg->finished_tasks))
		{
			_task_delete(_task);
		}

		release_fifo(worker_arg->finished_tasks);
	}
}
