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
