/*
 * @file timing.c
 *
 * @author Milosz Ciznicki 
 * @date 09-05-2011
 */

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "timing.h"
#include "../schedulers/scheduler.h"
#include "../queues/fifo_queue.h"

#ifndef CLOCK_MONOTONIC_RAW
#define CLOCK_MONOTONIC_RAW 4
#endif

void print_ntime(struct timespec *ts);
void print_utime(struct timespec *ts);
void print_stime(struct timespec *ts);

/* Reference starting global time */
static struct timespec global_start_time;

void _clock_gettime(struct timespec *ts)
{
	clock_gettime(CLOCK_MONOTONIC_RAW, ts);
	//printf("%ld %ld\n", ts->tv_sec, ts->tv_nsec);
}

void timespec_sub(struct timespec *ts_a, struct timespec *ts_b, struct timespec *ts_result)
{
	ts_result->tv_sec = ts_a->tv_sec - ts_b->tv_sec;
	ts_result->tv_nsec = ts_a->tv_nsec - ts_b->tv_nsec;

	if (ts_result->tv_nsec < 0)
	{
		ts_result->tv_sec -= 1;
		ts_result->tv_nsec += 1000000000;
	}
}

void timespec_acc(struct timespec *ts_result, struct timespec *ts_a)
{
	ts_result->tv_sec += ts_a->tv_sec;
	ts_result->tv_nsec += ts_a->tv_nsec;

	if (ts_result->tv_nsec >= 1000000000)
	{
		ts_result->tv_sec += 1;
		ts_result->tv_nsec -= 1000000000;
	}
}

static int8_t timespec_comp(struct timespec *ts_a, struct timespec *ts_b)
{
	struct timespec ts_diff;

	timespec_sub(ts_a, ts_b, &ts_diff);

	if (ts_diff.tv_sec > 0)
	{
		return 1;
	} else if (ts_diff.tv_sec < 0)
	{
		return -1;
	} else
	{
		if (ts_diff.tv_nsec > 0)
		{
			return 1;
		} else if (ts_diff.tv_nsec == 0)
		{
			return 0;
		}
	}

	return 0;
}

struct timespec *timespec_max(struct timespec *ts_a, struct timespec *ts_b)
{
	if (timespec_comp(ts_a, ts_b) == 1)
	{
		return ts_a;
	} else if (timespec_comp(ts_a, ts_b) == -1)
	{
		return ts_b;
	} else
	{
		return ts_a;
	}
}

struct timespec *timespec_min(struct timespec *ts_a, struct timespec *ts_b)
{
	if (timespec_comp(ts_a, ts_b) == 1)
	{
		return ts_b;
	} else if (timespec_comp(ts_a, ts_b) == -1)
	{
		return ts_a;
	} else
	{
		return ts_a;
	}
}

void timespec_asg(struct timespec *ts_dst, struct timespec *ts_src)
{
	ts_dst->tv_sec = ts_src->tv_sec;
	ts_dst->tv_nsec = ts_src->tv_nsec;
}

void update_time(struct timespec *ts_dst, struct timespec *ts_start, struct timespec *ts_end)
{
	struct timespec elapsed_time;

	timespec_sub(ts_start, ts_end, &elapsed_time);

	timespec_acc(ts_dst, &elapsed_time);
}

void get_relative_time(struct timespec *ts)
{
	struct timespec curr_ts;

	_clock_gettime(&curr_ts);

	//printf("%ld %ld\n", curr_ts.tv_sec, curr_ts.tv_nsec);

	timespec_sub(&curr_ts, &global_start_time, ts);

	//print_stime(ts);
	//printf("\n");
}

int64_t get_relative_time_us()
{
	struct timespec ts_curr;

	get_relative_time(&ts_curr);

	return (ts_curr.tv_sec * 1000000) + (int64_t) ceilf(ts_curr.tv_nsec / 1000.0);
}

int64_t get_relative_time_ms()
{
	struct timespec ts_curr;

	get_relative_time(&ts_curr);

	return (ts_curr.tv_sec * 1000) + (int64_t) ceilf(ts_curr.tv_nsec / 1000000.0);
}

void init_global_time()
{
	_clock_gettime(&global_start_time);
}

void timespec_clear(struct timespec *ts)
{
	ts->tv_sec = 0;
	ts->tv_nsec = 0;
}

task_timing *init_task_timing()
{
	task_timing *tk_t;

	tk_t = (task_timing *) malloc(sizeof(task_timing));

	timespec_clear(&tk_t->start_time);
	timespec_clear(&tk_t->end_time);
	timespec_clear(&tk_t->submit_time);

	tk_t->worker_id = -1;

	return tk_t;
}

void deinit_task_timing(task_timing *tk_t)
{
	free(tk_t);
}

worker_timing *init_worker_timing()
{
	worker_timing *wk_t;
	wk_t = (worker_timing *) malloc(sizeof(worker_timing));

	timespec_clear(&wk_t->start_time);
	timespec_clear(&wk_t->running_time);
	timespec_clear(&wk_t->executing_time);
	timespec_clear(&wk_t->idle_time);

	wk_t->executed_tasks = 0;

	return wk_t;
}

void deinit_worker_timing(worker_timing *wk_t)
{
	free(wk_t);
}

void print_ntime(struct timespec *ts)
{
	//	printf("%ld %ld;", ts->tv_sec, ts->tv_nsec);
	printf("%ld ", (ts->tv_sec * 1000000000) + ts->tv_nsec);
}

void print_utime(struct timespec *ts)
{
	printf("%10ld	", (ts->tv_sec * 1000000) + (long int) ceilf(ts->tv_nsec / 1000.0));
}

void print_stime(struct timespec *ts)
{
	printf("%10ld	", (ts->tv_sec * 1000) + (long int) ceilf(ts->tv_nsec / 1000000.0));
//	printf("%10ld	", ((ts->tv_sec * 1000) + (long int) ceilf(ts->tv_nsec / 1000000.0))/10);
}

long int get_stime(struct timespec *ts)
{
	return (ts->tv_sec * 1000) + (long int) ceilf(ts->tv_nsec / 1000000.0);
}

void print_time_length(struct timespec *ts_a, struct timespec *ts_b)
{
	struct timespec ts_length;

	timespec_sub(ts_a, ts_b, &ts_length);
	print_utime(&ts_length);
}

void get_timing_results()
{
	struct timespec ts_tmp;
	struct timespec *ts_a;
	struct timespec *ts_b;
	hs_config *config = get_config();

	int worker_id = 0;
	for (worker_id = 0; worker_id < config->nworkers; ++worker_id)
	{
		hs_worker *worker = &config->workers[worker_id];

		printf("%s\n", worker->description);
		//		print_utime(&global_start_time);

		ts_a = &ts_tmp;
		timespec_clear(ts_a);
		_task_t _task = fifo_pop_task(worker->finished_tasks);
		for (; _task != NULL; _task = fifo_pop_task(worker->finished_tasks))
		{
			hs_task *task = _task->task;
			task_timing *timing = task->timing;
			ts_b = &timing->start_time;
			print_time_length(ts_b, ts_a);
			ts_a = ts_b;
			ts_b = &timing->end_time;
			print_time_length(ts_b, ts_a);
			ts_a = ts_b;
			_task_delete(_task);
		}
		printf("\n");
	}
}

void get_timing_results_from_global_time()
{
	struct timespec ts_tmp;
	struct timespec *ts_a;
	struct timespec *ts_b;
	hs_config *config = get_config();
	int first = 1;

	int worker_id = 0;
	for (worker_id = 0; worker_id < config->nworkers; ++worker_id)
	{
		hs_worker *worker = &config->workers[worker_id];

		first = 1;
		timespec_clear(&ts_tmp);
		ts_a = &ts_tmp;
		_task_t _task = fifo_pop_task(worker->finished_tasks);
		for (; _task != NULL; _task = fifo_pop_task(worker->finished_tasks))
		{
			hs_task *task = _task->task;
			task_timing *timing = task->timing;

			/*if(first)
			 {
			 ts_a = &timing->start_time;
			 ts_b = &timing->start_time;
			 first = 0;
			 }*/

			printf("%s	", worker->description);
			//			print_time_length(ts_a, ts_b);
			print_utime(ts_a);
			ts_a = &timing->start_time;
			//			print_time_length(ts_a, ts_b);
			print_utime(ts_a);
			printf("sleep\n");

			printf("%s	", worker->description);
			ts_a = &timing->start_time;
			//			print_time_length(ts_a, ts_b);
			print_utime(ts_a);
			ts_a = &timing->end_time;
			//			print_time_length(ts_a, ts_b);
			print_utime(ts_a);
			printf("exec\n");
			_task_delete(_task);
		}
	}
}

void get_timing_results_from_exec_time()
{
	struct timespec *ts_tmp;
	struct timespec ts_zero;
	struct timespec *ts_a;
	struct timespec ts_b;
	double mean_flow_time = 0.0;
	hs_config *config = get_config();
	int first = 1;

	timespec_clear(&ts_zero);
	ts_tmp = &ts_zero;
	timespec_clear(&ts_b);

	int worker_id = 0;
	for (worker_id = 0; worker_id < config->nworkers; ++worker_id)
	{
		hs_worker *worker = &config->workers[worker_id];
		worker_timing *timing = worker->timing;

		if (worker->exectued == 1)
		{
			if (first)
			{
				ts_tmp = &timing->start_time;
				first = 0;
			}

			ts_tmp = timespec_min(ts_tmp, &timing->start_time);
		}
	}

	for (worker_id = 0; worker_id < config->nworkers; ++worker_id)
	{
		hs_worker *worker = &config->workers[worker_id];

		ts_a = ts_tmp;
		_task_t _task = fifo_pop_task(worker->finished_tasks);
		for (; _task != NULL; _task = fifo_pop_task(worker->finished_tasks))
		{
			hs_task *task = _task->task;
			task_timing *timing = task->timing;

//			printf("%s	", worker->description);
//			print_time_length(ts_a, ts_b);
//			//			print_utime(ts_a);
//			ts_a = &timing->start_time;
//			print_time_length(ts_a, ts_b);
//			//			print_utime(ts_a);
//			printf("sleep\n");

			printf("%s	", worker->description);
			ts_a = &timing->submit_time;
			print_stime(ts_a);
			ts_a = &timing->start_time;
//			print_time_length(ts_a, ts_b);
			print_stime(ts_a);
			ts_a = &timing->end_time;
			if(get_stime(&ts_b) < get_stime(ts_a))
			{
				ts_b.tv_sec = ts_a->tv_sec;
				ts_b.tv_nsec = ts_a->tv_nsec;
			}
//			print_time_length(ts_a, ts_b);
			print_stime(ts_a);
			printf("%d\n", _task->task->id);
			mean_flow_time += get_stime(&timing->end_time) - get_stime(&timing->submit_time);
			_task_delete(_task);
		}
	}
	printf("C_max:");
	print_stime(&ts_b);
	printf("\n");
	printf("F:\t%f\n", mean_flow_time/1000.0);
}
