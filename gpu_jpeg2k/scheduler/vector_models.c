/*
 * @file vector_models.c
 *
 * @author Milosz Ciznicki 
 * @date 23-05-2011
 */

#include "schedulers/scheduler.h"

static double cpu_cost(void *data_interface)
{
	return 1.0;
}

static double cuda_cost(void *data_interface)
{
	return 200.0;
}

hs_model *vector_create_model()
{
	hs_model *model = create_model();

	model->mutli_arch_model[HS_CPU_ID].task_cost = cpu_cost;
	model->mutli_arch_model[HS_CUDA_ID].task_cost = cuda_cost;
	model->type = HS_ARCH;

	return model;
}
