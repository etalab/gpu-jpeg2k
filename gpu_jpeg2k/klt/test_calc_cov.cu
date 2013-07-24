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
 * test_calc_cov.cu
 *
 *  Created on: Dec 1, 2011
 *      Author: miloszc
 */
#include "../types/image_types.h"
extern "C" {
#include "calc_cov.h"
#include "../misc/memory_management.cuh"
}


void test_calc_cov() {
	type_data *h_i = NULL;
	type_data *h_j = NULL;
	int m = 4;
	cuda_h_allocate_mem((void **)&h_i, m * sizeof(type_data));
	cuda_h_allocate_mem((void **)&h_j, m * sizeof(type_data));

	int k = 0;
	for(k = 0; k < m; ++k) {
		h_i[k] = k + 1;
		h_j[k] = k + 1;
	}

	type_data *d_i = NULL;
	type_data *d_j = NULL;
	cuda_d_allocate_mem((void **)&d_i, m * sizeof(type_data));
	cuda_d_allocate_mem((void **)&d_j, m * sizeof(type_data));
	cuda_memcpy_htd(h_i, d_i, m * sizeof(type_data));
	cuda_memcpy_htd(h_j, d_j, m * sizeof(type_data));

	int res = calc_cov(d_i, d_j, m);
	printf("cov: %f == 30\n", res);
}
