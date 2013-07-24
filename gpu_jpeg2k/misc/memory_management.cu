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
/**
 * @file memory_management.cu
 *
 * @brief CUDA memory management functions wrappers.
 *
 * This a collection of wrappers for CUDA MM that include debugging information logging and error checking.
 *
 * @author Miłosz Ciżnicki
 * @author Jakub Misiorny <misiorny@man.poznan.pl>
 */
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>


#include "cuda_errors.h"
extern "C" {
	#include "memory_management.cuh"
#include "../print_info/print_info.h"
}

void cuda_d_free(void *data)
{
	cudaFree(data);
	checkCUDAError("cuda_d_free");
}

void cuda_h_free(void *data) {
	cudaFreeHost(data);
//	checkCUDAError("cuda_h_free");
}

void cuda_set_device_flags() {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	cudaSetDeviceFlags(cudaDeviceMapHost);

	if (!prop.canMapHostMemory) {
		printf("[allocate_mem]: Cannot allocate host-device mapped memory. Exitting!\n");
		exit(0);
	}
}

/**
 * @brief Allocates host page locked memory.
 *
 * @param data Pointer to data.
 * @param mem_size How many bytes of memory to allocate
 * @return Pointer to allocated memory.
 */
void cuda_h_allocate_mem(void **data, uint64_t mem_size)
{
	//MAPPED memory test - FAILURE!
/*	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	cudaSetDeviceFlags(cudaDeviceMapHost);

	if (!prop.canMapHostMemory) {
		printf("[allocate_mem]: Cannot allocate host-device mapped memory. Exitting!\n");
		exit(0);
	}*/
	cudaHostAlloc(data, mem_size, cudaHostAllocMapped);

//	println_var(INFO, "allocating: %i [kB]\n", mem_size/1024);
	/*cudaHostAlloc(data, mem_size, cudaHostAllocPortable);*/

	checkCUDAError("cuda_h_allocate_mem");
}

/**
 * @brief Allocates memory on the device.
 *
 * @param data Pointer to data.
 * @param mem_size How many bytes of memory to allocate
 * @return Pointer to allocated memory.
 */
void cuda_d_allocate_mem(void **data, uint64_t mem_size)
{
//	println_var(INFO, "mem_size: %d", mem_size);
	cudaMalloc(data, mem_size);
	checkCUDAError("cuda_d_allocate_mem");
}

void cuda_memcpy_hth(void *src, void *dst, uint64_t size) {
	cudaMemcpy(dst, src, size, cudaMemcpyHostToHost);
	checkCUDAError("cuda_memcpy_hth");
}

void cuda_memcpy_htd(void *src, void *dst, uint64_t size) {
	cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice);
	checkCUDAError("cuda_memcpy_htd");
}

void cuda_memcpy_dtd(void *src, void *dst, uint64_t size) {
	cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
	checkCUDAError("cuda_memcpy_dtd");
}

void cuda_memcpy_dth(void *src, void *dst, uint64_t size) {
	cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
	checkCUDAError("cuda_memcpy_dth");
}

void cuda_memcpy2d_dtd(void *src, size_t src_width, void *dst, size_t dst_width, size_t width, size_t height) {
	cudaMemcpy2D(dst, dst_width, src, src_width, width, height, cudaMemcpyDeviceToDevice);
	checkCUDAError("cuda_memcpy2d_dtd");
}

void cuda_memcpy2d_dth(void *src, size_t src_width, void *dst, size_t dst_width, size_t width, size_t height) {
	cudaMemcpy2D(dst, dst_width, src, src_width, width, height, cudaMemcpyDeviceToHost);
	checkCUDAError("cuda_memcpy2d_dtd");
}

void cuda_memcpy2d_htd(void *src, size_t src_width, void *dst, size_t dst_width, size_t width, size_t height) {
	cudaMemcpy2D(dst, dst_width, src, src_width, width, height, cudaMemcpyHostToDevice);
	checkCUDAError("cuda_memcpy2d_dtd");
}

void cuda_d_memset(void *dst, int val, uint64_t mem_size) {
	cudaMemset(dst, val, mem_size);
	checkCUDAError("cuda_d_memset");
}

void cuda_set_device(int i) {
	cudaSetDevice(i);
	checkCUDAError("cuda_set_device");
}

void cuda_set_printf_limit(size_t memSize) {
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, memSize);
	checkCUDAError("cuda_set_printf_limit");
}
