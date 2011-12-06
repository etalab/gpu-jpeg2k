#ifndef ADJUST_H_
#define ADJUST_H_

#include "klt.h"

#define FORWARD 1

void adjust_pca_data_mm(type_data* transformationMatrix, uint8_t forward, type_data* input, type_data* output,
		int componentCount, int componentLength);
void adjust_pca_data_mv(type_data* transformationMatrix, uint8_t forward, type_data* input, type_data* output,
		int componentCount, int componentLength);
void test_cublas();

__global__ void readSampleSimple(type_data** data, int sample_num, type_data* sample);
__global__ void readSamples(type_data** data, int num_vecs, int len_vec, type_data* sample);
__global__ void writeSampleSimple(type_data** data, int sample_num, type_data* sample);
__global__ void writeSamples(type_data** data, int num_vecs, int len_vec, type_data* sample);
__global__ void writeSampleWithSum(type_data** data, int sample_num, type_data* sample, type_data* means);

#endif
