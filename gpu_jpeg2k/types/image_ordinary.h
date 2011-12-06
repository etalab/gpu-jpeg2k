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

