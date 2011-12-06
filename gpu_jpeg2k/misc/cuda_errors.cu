/**
 * @file cuda_errors.cu
 *
 * @author Jakub Misiorny <misiorny@man.poznan.pl>
 */

#include "cuda_errors.h"
#include <stdio.h>

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}
