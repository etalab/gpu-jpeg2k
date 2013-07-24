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
#ifndef __gpu_coder_h__
#define __gpu_coder_h__

#include "../../types/image_types.h"

#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif

#define MAX_CODESTREAM_SIZE (4096 * 10) /// TODO: figure out

#define LL_LH_SUBBAND	0
#define HL_SUBBAND		1
#define HH_SUBBAND		2

typedef unsigned char byte;

typedef struct
{
	int significantBits;
	int codingPasses;
	int *coefficients;
	int *h_coefficients;
	int subband;
	int width;
	int height;

	int nominalWidth;
	int nominalHeight;

	int magbits;
	int compType;
	int dwtLevel;
	float stepSize;

	byte *codeStream;
	int length;
	type_codeblock *cblk;
} EntropyCodingTaskInfo;

//#include "../../../misc/memory_management.cuh"

extern void encode_tile(type_tile *tile);
extern void encode_tile_dbg(type_tile *tile);
void perform_test(const char * test_file_path);
extern float gpuEncode(EntropyCodingTaskInfo *infos, int count, int targetSize);

extern void decode_tile(type_tile *tile);

#endif
