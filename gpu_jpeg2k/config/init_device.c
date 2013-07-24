/* 
Copyright 2009-2013 Poznan Supercomputing and Networking Center

Authors:
Milosz Ciznicki miloszc@man.poznan.pl

GPU JPEG2K is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GPU JPEG2K is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with GPU JPEG2K. If not, see <http://www.gnu.org/licenses/>.
*/
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

	cuda_set_device_flags();

	size_t printf_buff = 10 * 1024 * 1024 * 100;

	cuda_set_printf_limit(printf_buff);
	cuda_set_device(param->param_device);

	cuda_d_allocate_mem((void**)&d_tmp, sizeof(int));
	cuda_memcpy_htd(&h_tmp, d_tmp, sizeof(int));
	h_tmp = 2;
	cuda_memcpy_dth(d_tmp, &h_tmp, sizeof(int));
}
