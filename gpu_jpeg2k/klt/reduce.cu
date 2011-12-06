/*
 * reduce.cu
 *
 *  Created on: Nov 30, 2011
 *      Author: miloszc
 */
#include "../types/image_types.h"
extern "C" {
#include "reduce.h"
#include "blocks.h"
#include "../misc/memory_management.cuh"
}
#include "../misc/cuda_errors.h"

#define MAX_THREADS 128

static bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

/*
 * @brief Reduction kernel
 * @param g_idata	device input data
 * @param g_odata	device reduced output data
 * @param n			list size
 */
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce(T *g_idata, T *g_odata, unsigned int n)
{
	__shared__ T sdata[1024];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	T mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += g_idata[i];
		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			mySum += g_idata[i+blockSize];
		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	__syncthreads();

	// do reduction in shared mem
	if (blockSize >= 512) {if (tid < 256) {sdata[tid] = mySum = mySum + sdata[tid + 256];}__syncthreads();}
	if (blockSize >= 256) {if (tid < 128) {sdata[tid] = mySum = mySum + sdata[tid + 128];}__syncthreads();}
	if (blockSize >= 128) {if (tid < 64) {sdata[tid] = mySum = mySum + sdata[tid + 64];}__syncthreads();}

	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile T* smem = sdata;
		if (blockSize >= 64) {smem[tid] = mySum = mySum + smem[tid + 32];}
		if (blockSize >= 32) {smem[tid] = mySum = mySum + smem[tid + 16];}
		if (blockSize >= 16) {smem[tid] = mySum = mySum + smem[tid + 8];}
		if (blockSize >= 8) {smem[tid] = mySum = mySum + smem[tid + 4];}
		if (blockSize >= 4) {smem[tid] = mySum = mySum + smem[tid + 2];}
		if (blockSize >= 2) {smem[tid] = mySum = mySum + smem[tid + 1];}
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

/*
 * @brief Execute kernel with appropriate number of threads.
 */
void graph_reduce(int threads, int blocks, type_data *d_idata, type_data *d_odata, int size) {
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

	if (isPow2(size)) {
		switch (threads) {
		case 512:
			reduce<type_data, 512, true><<< dimGrid, dimBlock>>>(d_idata, d_odata, size); break;
		case 256:
			reduce<type_data, 256, true><<< dimGrid, dimBlock>>>(d_idata, d_odata, size); break;
		case 128:
			reduce<type_data, 128, true><<< dimGrid, dimBlock>>>(d_idata, d_odata, size); break;
		case 64:
			reduce<type_data, 64, true><<< dimGrid, dimBlock>>>(d_idata, d_odata, size); break;
		case 32:
			reduce<type_data, 32, true><<< dimGrid, dimBlock>>>(d_idata, d_odata, size); break;
		case 16:
			reduce<type_data, 16, true><<< dimGrid, dimBlock>>>(d_idata, d_odata, size); break;
		case 8:
			reduce<type_data, 8, true><<< dimGrid, dimBlock>>>(d_idata, d_odata, size); break;
		case 4:
			reduce<type_data, 4, true><<< dimGrid, dimBlock>>>(d_idata, d_odata, size); break;
		case 2:
			reduce<type_data, 2, true><<< dimGrid, dimBlock>>>(d_idata, d_odata, size); break;
		case 1:
			reduce<type_data, 1, true><<< dimGrid, dimBlock>>>(d_idata, d_odata, size); break;
		}
	} else {
		switch (threads) {
		case 512:
			reduce<type_data, 512, false><<< dimGrid, dimBlock>>>(d_idata, d_odata, size); break;
		case 256:
			reduce<type_data, 256, false><<< dimGrid, dimBlock>>>(d_idata, d_odata, size); break;
		case 128:
			reduce<type_data, 128, false><<< dimGrid, dimBlock>>>(d_idata, d_odata, size); break;
		case 64:
			reduce<type_data, 64, false><<< dimGrid, dimBlock>>>(d_idata, d_odata, size); break;
		case 32:
			reduce<type_data, 32, false><<< dimGrid, dimBlock>>>(d_idata, d_odata, size); break;
		case 16:
			reduce<type_data, 16, false><<< dimGrid, dimBlock>>>(d_idata, d_odata, size); break;
		case 8:
			reduce<type_data, 8, false><<< dimGrid, dimBlock>>>(d_idata, d_odata, size); break;
		case 4:
			reduce<type_data, 4, false><<< dimGrid, dimBlock>>>(d_idata, d_odata, size); break;
		case 2:
			reduce<type_data, 2, false><<< dimGrid, dimBlock>>>(d_idata, d_odata, size); break;
		case 1:
			reduce<type_data, 1, false><<< dimGrid, dimBlock>>>(d_idata, d_odata, size); break;
		}
	}
}

/*
 * @brief Reduce graph data.
 * @param device graph input data
 * @param width and height of graph data
 * @return device reduced data
 */
type_data reduction(type_data *d_i_data, type_data *d_o_data, type_data *h_odata, int size) {
	int maxThreads = MAX_THREADS; /* number of threads per block */
	int maxBlocks = MAX_BLOCKS; /* maximum number of blocks used for reducition */
	int numBlocks = 0; /* default number of blocks used for reducing one list */
	int numThreads = 0; /* default number of threads */
	dim3 threads, blocks;
	type_data gpu_result = 0.0f;

	/* calculate how many blocks and threads use for reduction kernel */
	get_num_blocks_and_threads(size, maxBlocks, maxThreads, &numBlocks, &numThreads);

//	printf("%d, %d, %d, %d, %d\n", size, maxBlocks, maxThreads, numBlocks, numThreads);

	/* sum up all */
	graph_reduce(numThreads, numBlocks, d_i_data, d_o_data, size);
	checkCUDAError("\tafter reduce");

	cuda_memcpy_dth(d_o_data, h_odata, numBlocks * sizeof(type_data));

	for(int i=0; i<numBlocks; i++)
	{
		gpu_result += h_odata[i];
	}

	return gpu_result;
}
