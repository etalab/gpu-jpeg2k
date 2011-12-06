/*
 * @file common_worker.c
 *
 * @author Milosz Ciznicki 
 * @date 09-05-2011
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <time.h>
#include <unistd.h>
#include "worker_helper.h"
#include "../timing/timing.h"

void sleep_worker(hs_worker *worker)
{
	struct timespec start_time, end_time;
	worker_timing *timing = worker->timing;

	get_relative_time(&start_time);

	cond_wait_queue(worker->task_queue, worker->arch);

	get_relative_time(&end_time);

	update_time(&timing->idle_time, &start_time, &end_time);
}

void update_worker_exec_status(worker_timing *timing, struct timespec *ts_start, struct timespec *ts_end)
{
	timing->executed_tasks += 1;

	update_time(&timing->executing_time, ts_start, ts_end);
}
