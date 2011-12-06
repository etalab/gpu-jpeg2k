/*
 * @file test.c
 *
 * @author Milosz Ciznicki
 * @date 06-05-2011
 */

#include <pthread.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>

#include "schedulers/scheduler.h"
#include "test.h"
#include "test_models.h"
#include "read_file.h"

extern void test_cpu_func(void *data_interface);
extern void test_cuda_func(void *data_interface);

static void deinit_test_data(hs_task *task)
{
	data *data_i = (data *)task->data_interface;

	free(data_i);
	destroy_task(task);
}

static void validate_test_data(hs_task *task)
{
	data *data_i = (data *)task->data_interface;
}

int main(int argc, char **argv)
{
	init_scheduler();

	int ntasks = atoi(argv[1]);
	hs_task **tasks = (hs_task **) malloc(ntasks * sizeof(hs_task *));

	if(argc < 3)
	{
		printf("Provide number of tasks and file name!\n");
		return 0;
	}

	printf("%d\n", ntasks);

	int **e = get_execution_times(argv[2]);

	if(e == NULL)
	{
		printf("Empty e array!\n");
		return 0;
	}

	int i;
	for(i = 0; i < ntasks; ++i)
	{
		tasks[i] = create_task();

		data *data_i = (data *) malloc(sizeof(data));
		data_i->cpu_time = e[i][0];
		data_i->gpu_time = e[i][1];

		tasks[i]->id = i;
		tasks[i]->data_interface = data_i;
		tasks[i]->model = test_create_model();

		tasks[i]->arch_type = HS_ARCH_CPU|HS_ARCH_CUDA;
		tasks[i]->cpu_func = test_cpu_func;
		tasks[i]->cuda_func = test_cuda_func;
		tasks[i]->cost_interface = (void *)data_i;

		submit_task(tasks[i]);
		usleep(e[i][2]*1000);
	}

	synchronize_tasks();

	get_timing_results_from_exec_time();

	shutdown_scheduler();

	for(i = 0; i < ntasks; ++i)
	{
		validate_test_data(tasks[i]);
		deinit_test_data(tasks[i]);
	}

	free(tasks);

	pthread_exit(NULL);

	return 0;
}
