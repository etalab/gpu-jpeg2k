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
 * @brief Header part of the preprocess_gpu.cu
 *
 * @author Jakub Misiorny <misiorny@man.poznan.pl>
 */


#ifndef PREPROCESS_GPU_H_
#define PREPROCESS_GPU_H_

#include "../types/image_types.h"
//#define COMPUTE_TIME 1

/* cuda block is a square with a side of BLOCK_SIZE. Actual number of threads in the block is the square of this value*/
#define BLOCK_SIZE 16
#define TILE_SIZEX 32
#define TILE_SIZEY 32

typedef enum {
	RCT, ///Reversible Color Transformation. Encoder part of the lossless flow.
	TCR, ///Decoder part of the lossless flow.
	ICT, ///Irreversible Color Transformation. Encoder part of the lossy flow.
	TCI  ///Decoder part of the lossy flow.
} color_trans_type;


extern int color_trans_gpu(type_image *img, color_trans_type type);

int color_coder_lossy(type_image *img);
int color_decoder_lossy(type_image *img);
int color_coder_lossless(type_image *img);
int color_decoder_lossless(type_image *img);
void fdc_level_shifting(type_image *img);
void idc_level_shifting(type_image *img);

#endif
