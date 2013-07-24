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
 * @file gs.cu
 *
 * @author Kamil Balwierz
 */

#include <math.h>
#include <time.h>
#include "gs.h"
#include <cublas.h>


int gram_schmidt(int N, type_data* output, type_data *dinput, type_data *eValues, int J, type_data er) {
	cublasStatus status;
	int j, k;
	type_data *dT = 0;
	status = cublasAlloc(N*N, sizeof (type_data), (void**) &dT);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "! device memory allocation error (dT)\n");
		return EXIT_FAILURE;
	}
	type_data *doutput = 0;
	status = cublasAlloc(N*N, sizeof (type_data), (void**) &doutput);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "! device memory allocation error (doutput)\n");
		return EXIT_FAILURE;
	}
	if (eValues == 0) {
		fprintf(stderr, "! host memory allocation error: T\n");
		return EXIT_FAILURE;
	}
	type_data *dU = 0;
	status = cublasAlloc(N, sizeof (type_data), (void**) &dU);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "! device memory allocation error (dU)\n");
		return EXIT_FAILURE;
	}

	int iter = 0;
	type_data a;
	for (k = 0; k < N; k++) {
		cublasScopy(N, &dinput[k * N], 1, &dT[k * N], 1);
		a = 0.0;
		for (j = 0; j < J; j++) {
			cublasSgemv('t', N, N, 1.0f, dinput, N, &dT[k * N], 1, 0.0f, &doutput[k * N], 1);
			if (k > 0) {
				cublasSgemv('t', N, k, 1.0f, doutput, N, &doutput[k * N], 1, 0.0f, dU, 1);
				cublasSgemv('n', N, k, -1.0f, doutput, N, dU, 1, 1.0f, &doutput[k * N], 1);
			}
			cublasSscal(N, 1.0f / cublasSnrm2(N, &doutput[k * N], 1), &doutput[k * N], 1);
			cublasSgemv('n', N, N, 1.0f, dinput, N, &doutput[k * N], 1, 0.0f, &dT[k * N], 1);
			if (k > 0) {
				cublasSgemv('t', N, k, 1.0f, dT, N, &dT[k * N], 1, 0.0, dU, 1);
				cublasSgemv('n', N, k, -1.0f, dT, N, dU, 1, 1.0f, &dT[k * N], 1);
			}
			eValues[k] = cublasSnrm2(N, &dT[k * N], 1);
			cublasSscal(N, 1.0f / eValues[k], &dT[k * N], 1);
//			printf("%f\n", fabs(a - eValues[k]));
			if (fabs(a - eValues[k]) < er * eValues[k]) {
//				printf("iterations %d\n", j);
				break;
			}
			a = eValues[k];
			iter++;
		}
//		iter = 0;
		cublasSger(N, N, -eValues[k], &dT[k * N], 1, &doutput[k * N], 1, dinput, N);
	}
	printf("iter %d\n", iter);
	for (k = 0; k < N; k++) {
		cublasSscal(N, eValues[k], &dT[k * N], 1);
	}
//	cublasSgemm('n', 'n', N, N, N, 1.0, dT, N, doutput, N, 0.0, output, N);
	cublasGetMatrix(N, N, sizeof (doutput[0]), doutput, N, output, N);
	status = cublasFree(doutput);
	status = cublasFree(dT);
	return EXIT_SUCCESS;
}
