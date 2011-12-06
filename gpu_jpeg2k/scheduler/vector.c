/*
 * @file vector.c
 *
 * @author Milosz Ciznicki 
 * @date 06-05-2011
 */

#include <pthread.h>
#include <stdio.h>
#include <assert.h>

#include "schedulers/scheduler.h"
#include "vector.h"
#include "vector_models.h"

extern void scal_cpu_func(void *data_interface);
extern void scal_cuda_func(void *data_interface);

static void init_vector_data(hs_task *task)
{
	vector *vec = (vector *) malloc(sizeof(vector));

	vec->size = 3000;
	vec->array = (float *) malloc(vec->size * sizeof(float));

	int i;
	for(i = 0; i < vec->size; ++i)
	{
		vec->array[i] = 1;
	}

	data *data_i = (data *) malloc(sizeof(data));
	data_i->vec = vec;
	data_i->factor = 3;

	task->data_interface = data_i;
	task->model = vector_create_model();
}

static void deinit_vector_data(hs_task *task)
{
	data *data_i = (data *)task->data_interface;

	free(data_i->vec->array);
	free(data_i->vec);
	free(data_i);
	destroy_task(task);
}

static void validate_vector_data(hs_task *task)
{
	data *data_i = (data *)task->data_interface;

	int i;
	for(i = 0; i < data_i->vec->size; ++i)
	{
//		printf("%f\n", data_i->vec->array[i]);
		assert(data_i->vec->array[i] == 3);
	}
}

int main(int argc, char **argv)
{
	init_scheduler();

	int ntasks = 199;
	hs_task **tasks = (hs_task **) malloc(ntasks * sizeof(hs_task *));

	int i;
	for(i = 0; i < ntasks; ++i)
	{
		tasks[i] = create_task();

		init_vector_data(tasks[i]);

		tasks[i]->arch_type = HS_ARCH_CPU|HS_ARCH_CUDA;
		tasks[i]->cpu_func = scal_cpu_func;
		tasks[i]->cuda_func = scal_cuda_func;

		submit_task(tasks[i]);
	}

	synchronize_tasks();

	get_timing_results_from_exec_time();

	shutdown_scheduler();

	for(i = 0; i < ntasks; ++i)
	{
		validate_vector_data(tasks[i]);
		deinit_vector_data(tasks[i]);
	}

	free(tasks);

	pthread_exit(NULL);

	return 0;
}
