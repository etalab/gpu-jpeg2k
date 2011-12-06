/*
 * @file vector_cuda.cu
 *
 * @author Milosz Ciznicki 
 * @date 06-05-2011
 */

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "vector.h"

static __global__ void vector_mult_cuda(float *val, unsigned n,
		float factor)
{
	unsigned i;
	for(i = 0; i < n; i++)
		val[i] *= factor;
}

extern "C" void scal_cuda_func(void *data_interface)
{
	data *data_i = (data *)data_interface;
	vector *vec = data_i->vec;
	float factor = data_i->factor;

	/* length of the vector */
	unsigned n = vec->size;

	float *d_array;

	cudaMalloc((void **)&d_array, n * sizeof(float));

	cudaMemcpy(d_array, vec->array, n * sizeof(float), cudaMemcpyHostToDevice);

	/* TODO: use more blocks and threads in blocks */
	vector_mult_cuda<<<1,1>>>(d_array, n, factor);

	cudaMemcpy(vec->array, d_array, n * sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
}
