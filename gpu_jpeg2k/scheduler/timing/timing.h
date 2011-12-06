/*
 * @file profiling.h
 *
 * @author Milosz Ciznicki 
 * @date 09-05-2011
 */

#ifndef PROFILING_H_
#define PROFILING_H_

#include <time.h>
#include <stdint.h>

typedef struct task_timing_t {
	struct timespec start_time;
	struct timespec end_time;
	struct timespec submit_time;
	int32_t worker_id;
} task_timing;

typedef struct worker_timing_t {
	struct timespec start_time;
	struct timespec running_time;
	struct timespec executing_time;
	struct timespec idle_time;
	int32_t executed_tasks;
} worker_timing;

void init_global_time();
void timespec_clear(struct timespec *ts);
void get_relative_time(struct timespec *ts);
int64_t get_relative_time_ms();
task_timing *init_task_timing();
void deinit_task_timing(task_timing *tk_t);
worker_timing *init_worker_timing();
void deinit_worker_timing(worker_timing *wk_t);
void timespec_acc(struct timespec *ts_result, struct timespec *ts_a);
void timespec_sub(struct timespec *ts_a, struct timespec *ts_b, struct timespec *ts_result);
struct timespec *timespec_max(struct timespec *ts_a, struct timespec *ts_b);
void timespec_asg(struct timespec *ts_dst, struct timespec *ts_src);
int64_t get_relative_time_us();
void update_time(struct timespec *ts_dst, struct timespec *ts_start, struct timespec *ts_end);
void get_timing_results_from_global_time();
void get_timing_results_from_exec_time();

#endif /* PROFILING_H_ */
