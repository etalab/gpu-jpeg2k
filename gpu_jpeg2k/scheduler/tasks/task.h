/*
 * @file task.h
 *
 * @author Milosz Ciznicki 
 * @date 06-05-2011
 */

#ifndef TASK_H_
#define TASK_H_

#include "../timing/timing.h"
#include "../models/cost_model.h"

typedef struct hs_task_t {
	int16_t id;
	int8_t arch_type;
	void (*cpu_func)(void *);
	void (*cuda_func)(void *);
	void *data_interface;
	hs_model *model;
	void *cost_interface;
	task_timing *timing;
} hs_task;

hs_task *create_task();
void destroy_task(hs_task *task);
void submit_task(hs_task *task);
void dec_nsubmitted_tasks();
void synchronize_tasks();

#endif /* TASK_H_ */
