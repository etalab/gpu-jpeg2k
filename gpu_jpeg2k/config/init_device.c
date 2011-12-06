/*
 * init_device.c
 *
 *  Created on: Dec 1, 2011
 *      Author: miloszc
 */

#include "init_device.h"
#include "../misc/memory_management.cuh"

void init_device(type_parameters *param)
{
	int *d_tmp;
	int h_tmp = 1;

	int printf_buff = 10 * 1024 * 1024;

//	cuda_set_printf_limit(printf_buff);
	cuda_set_device(param->param_device);

	cuda_d_allocate_mem((void**)&d_tmp, sizeof(int));
	cuda_memcpy_htd(&h_tmp, d_tmp, sizeof(int));
	h_tmp = 2;
	cuda_memcpy_dth(d_tmp, &h_tmp, sizeof(int));
}
