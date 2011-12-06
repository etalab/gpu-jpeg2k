/*
 * cuda_errors.h
 *
 *  Created on: Sep 28, 2010
 *      Author: qba
 */

#ifndef CUDA_ERRORS_H_
#define CUDA_ERRORS_H_

#include <cuda.h>
#include <cuda_runtime_api.h>

void checkCUDAError(const char *msg);

#endif /* CUDA_ERRORS_H_ */
