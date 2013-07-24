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
