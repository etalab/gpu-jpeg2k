/**
 * @file image_hyper.h
 *
 * @author Milosz Ciznicki
 */

#ifndef IMAGE_HYPER_H_
#define IMAGE_HYPER_H_

#include "image_types.h"
#include "../config/parameters.h"

typedef struct type_image_hyper type_image_hyper;

/** Image parameters */
struct type_image_hyper {
	int num_samples;//Numero de samples
	int num_lines;//Numero de lineas
	int num_bands;//Numero de bandas
	int data_type;//Tipo de datos
	long int lines_samples;// num_lines*num_samples
};

void read_header(const char *filename_header, type_image_hyper *image);
int read_hyper_image(type_image **_container, type_parameters *param);
void save_img_hyper(type_image *img, const char* out_file);
void write_imge(float *data, char *filename, type_image_hyper *image);
void write_one_band(float *data, char *filename, type_image_hyper *image);

#endif /* IMAGE_HYPER_H_ */
