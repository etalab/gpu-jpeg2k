/**
 * @file image.h
 *
 * @author Milosz Ciznicki
 */

#ifndef IMAGE_H_
#define IMAGE_H_

void init_tiles(type_image **_img, type_parameters *param);
void set_coding_parameters(type_image *img, type_parameters *param);
int read_image(type_image *img, type_parameters *param);
void save_image(type_image *img);

#endif /* IMAGE_H_ */
