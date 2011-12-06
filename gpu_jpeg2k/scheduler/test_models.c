/**
 * @file test_model.c
 *
 * @author Milosz Ciznicki
 */

#include "schedulers/scheduler.h"
#include "test.h"

static double cpu_cost(void *cost_interface)
{
	data *data_i = (data *)cost_interface;
	return data_i->cpu_time * 1000;
}

static double cuda_cost(void *cost_interface)
{
	data *data_i = (data *)cost_interface;
	return data_i->gpu_time * 1000;
}

hs_model *test_create_model()
{
	hs_model *model = create_model();

	model->mutli_arch_model[HS_CPU_ID].task_cost = cpu_cost;
	model->mutli_arch_model[HS_CUDA_ID].task_cost = cuda_cost;
	model->type = HS_ARCH;

	return model;
}
