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
 * @file image.h
 *
 * @author Milosz Ciznicki
 * @author Jakub Misiorny
 */


#ifndef READ_IMAGE_H_
#define READ_IMAGE_H_

#include "../config/parameters.h"
#include "image_types.h"
#include <FreeImage.h>

#define TILE_SIZE 1024

long int read_ordinary_image(type_image **img, type_parameters *param);
int save_img_ord(type_image *img, const char *filename);
int save_img_grayscale(type_image *img, char *filename);
void save_tile_comp(type_tile_comp *tile_comp, char *filename);
void init_device();
uint8_t get_num_comp(FIBITMAP* dib);
uint8_t get_img_type(FIBITMAP* dib);

void print_image(type_image *img);
#endif /* READ_IMAGE_H_ */

