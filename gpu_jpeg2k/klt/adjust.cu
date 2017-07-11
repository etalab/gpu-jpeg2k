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
#include "adjust.h"
#include <cublas.h>
extern "C" {
#include "../misc/memory_management.cuh"
#include "../print_info/print_info.h"
}
/*
 * @brief C = alpha * op(A) * op(B) + beta * C
 *
 * Matrices A,B and C stored in column-major format.
 */
void adjust_pca_data_mm(type_data* transformationMatrix, uint8_t forward, type_data* input, type_data* output,
		int componentCount, int componentLength) {
	// If 'y' op(A)=A^T, else op(A)=A
	char transa = (forward == FORWARD) ? 't' : 'n';
	// If 'y' op(B)=B^T, else op(B)=B
	char transb = 'n';
	int m = (forward == FORWARD) ? componentCount : componentLength;
	int n = 1;
	int k = (forward == FORWARD) ? componentLength : componentCount;
	// alpha * op(A) * op(B)
	float aplha = 1.0f;
	// m x k
	const float *A = transformationMatrix;
	// leading dimension
	int lda = componentLength;
	// k x n
	const float *B = input;
	int ldb = (forward == FORWARD) ? componentLength : componentCount;
	float beta = 0.0f;
	// m x n
	float *C = output;
	int ldc = (forward == FORWARD) ? componentCount : componentLength;

	cublasSgemm(transa, transb, m, n, k, aplha, A, lda, B, ldb, beta, C, ldc);
//	cublasSgemm((forward == FORWARD) ? 't' : 'n', 'n', (forward == FORWARD) ? componentCount : componentLength, 1,
//			(forward == FORWARD) ? componentLength : componentCount, 1, transformationMatrix, componentLength, input,
//			(forward == FORWARD) ? componentLength : componentCount, 0, output,
//			(forward == FORWARD) ? componentCount : componentLength);
}

/*
 * @brief y = alpha * op(A) * x + beta * y
 *
 * Matrix A.
 */
void adjust_pca_data_mv(type_data* transformationMatrix, uint8_t forward, type_data* input, type_data* output,
		int componentCount, int componentLength) {
	// If 'y' op(A)=A^T, else op(A)=A
	char transa = (forward == FORWARD) ? 't' : 'n';
	int m = (forward == FORWARD) ? componentCount : componentLength;
	int n = (forward == FORWARD) ? componentLength : componentCount;
	// alpha * op(A) * op(B)
	float aplha = 1.0f;
	// m x k
	const float *A = transformationMatrix;
	// leading dimension
	int lda = componentLength;
	// k x n
	const float *x = input;
	int incx = 1;
	float beta = 0.0f;
	// m x n
	float *y = output;
	int incy = 1;

	cublasSgemv(transa, m, n, aplha, A, lda, x, incx, beta, y, incy);
//	cublasSgemm((forward == FORWARD) ? 't' : 'n', 'n', (forward == FORWARD) ? componentCount : componentLength, 1,
//			(forward == FORWARD) ? componentLength : componentCount, 1, transformationMatrix, componentLength, input,
//			(forward == FORWARD) ? componentLength : componentCount, 0, output,
//			(forward == FORWARD) ? componentCount : componentLength);
}

void test_cublas() {
	int size = 2;
	int i = 0;
	// If 'y' op(A)=A^T, else op(A)=A
	char transa = 't';
	int m = size;
	int n = size;
	// alpha * op(A) * op(B)
	float aplha = 1.0f;

	float mat[] = {1.0f,2.0f,3.0f,4.0f};
	float *d_mat = NULL;
	cuda_d_allocate_mem((void **)&d_mat, size * size * sizeof(float));
	cuda_memcpy_htd(mat, d_mat, size * size * sizeof(float));

	// m x k
	const float *A = d_mat;
	// leading dimension
	int lda = size;
	float input[size];
	for(i = 0; i < size; ++i) {
		input[i] = i + 1;
	}

	float *d_input = NULL;
	cuda_d_allocate_mem((void **)&d_input, size * sizeof(float));
	cuda_memcpy_htd(input, d_input, size * sizeof(float));

	// k x n
	const float *x = d_input;
	int incx = 1;
	float beta = 0.0f;
	float output[size];
	for(i = 0; i < size; ++i) {
		output[i] = 0.0f;
	}
	float *d_output = NULL;
	cuda_d_allocate_mem((void **)&d_output, size * sizeof(float));
	cuda_memcpy_htd(output, d_output, size * sizeof(float));

	// m x n
	float *y = d_output;
	int incy = 1;

	cublasSgemv(transa, m, n, aplha, A, lda, x, incx, beta, y, incy);

	cuda_memcpy_dth(d_output, output, size * sizeof(float));

	for(i = 0; i < size; ++i) {
		printf("%f ", output[i]);
	}
	printf("\n");
}

__global__ void readSampleSimple(type_data** data, int sample_num, type_data* sample) {
	sample[threadIdx.x] = data[threadIdx.x][sample_num];
}

__global__ void readSamples(type_data** data, int num_vecs, int len_vec, type_data* sample) {
	int threadId = threadIdx.x + blockIdx.x * blockDim.x;
	if(threadId >= num_vecs)
		return;

	int i = 0;
	for(i = 0; i < len_vec; ++i) {
		sample[threadId * len_vec + i] = data[i][threadId];
	}
}

__global__ void writeSampleSimple(type_data** data, int sample_num, type_data* sample) {
	data[threadIdx.x][sample_num] = sample[threadIdx.x];
}

__global__ void writeSamples(type_data** data, int num_vecs, int len_vec, type_data* sample) {
	int threadId = threadIdx.x + blockIdx.x * blockDim.x;
	if(threadId >= num_vecs)
		return;

	int i = 0;
	for(i = 0; i < len_vec; ++i) {
		 data[i][threadId] = sample[threadId * len_vec + i];
	}
}

__global__ void writeSampleWithSum(type_data** data, int sample_num, type_data* sample, type_data* means) {
	data[threadIdx.x][sample_num] = sample[threadIdx.x] + means[threadIdx.x];
}
