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
#include "analysis.h"
#include <stdlib.h>

__device__ type_data calculate_mean(type_data* data, unsigned int count) {
	type_data sum = 0.0;
	int j;
	for(j=0; j<count; ++j) {
		sum += data[j];
	}
	for(j=0; j<count; ++j) {
		data[j] -= (type_data)sum/(type_data)count;
	}
	return sum/(type_data)count;
}


__global__ void mean_adjust_data_g(type_data** data, type_data* means, int count) {
	means[threadIdx.x] = calculate_mean(data[threadIdx.x], count);
}

__device__ type_data calculate_covariance(type_data* firstComponent, type_data* secondComponent, int count) {
	type_data sum = 0.0;
	for(int j=0; j<count; ++j) {
		sum += firstComponent[j]*secondComponent[j];
	}
	return sum/(type_data)count;
}

__global__ void calculate_covariance_matrix(type_data** data, type_data* covMatrix, int count, int dim) {
	int i = threadIdx.x;
	int j = blockIdx.x;
	if(j<i) return;

	covMatrix[i * dim + j] = calculate_covariance(data[i], data[j], count);
	covMatrix[j * dim + i] = covMatrix[i * dim + j];
}

__global__ void mean_adjust_data_new(type_data** data, type_data* means, int count) {
	means[threadIdx.x] = calculate_mean(data[threadIdx.x], count);
}
