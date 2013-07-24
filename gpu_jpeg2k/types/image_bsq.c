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
 * @file image_bsq.c
 *
 * @author Milosz Ciznicki
 */
#include "../types/image_bsq.h"
#include "../print_info/print_info.h"
#include <FreeImage.h>

/**
 * @brief Initializes image container
 *
 * @param container Malloc'ed type_image to be initialized.
 * @param image_bsq Image properties.
 */
void init_bsq_image(type_image *container, type_image_hyper *image_bsq, type_parameters *param)
{
	//	println_start(INFO);
	container->height = image_bsq->num_lines;
	container->width = image_bsq->num_samples;
	container->depth = (image_bsq->data_type == 1 ? 8 : 16) * image_bsq->num_bands;
	container->num_components = image_bsq->num_bands;
	container->num_range_bits = container->depth / container->num_components;
	printf("%d %d %d %d %d\n", container->height, container->width, container->depth, container->num_components, container->num_range_bits);

	/* TODO: check if origin data was signed? */
	container->sign = SIGNED;
	container->num_dlvls = param->param_tile_comp_dlvls;

	/* Set coding parameters */
	set_coding_parameters(container, param);
	init_tiles(&container, param);

	//	println_end(INFO);
}

/**
 * @brief Loads the hyperspectral data.
 *
 * @param image_filename
 * @param image_bsq
 * @return
 */
float *load_image(const char *image_filename, type_image_hyper *image_bsq)
{
	FILE *fp;
	short *type_short_int;
	double *type_double;
	float *type_float;
	float *h_imagen;
	int i;

	h_imagen = (float*) malloc(image_bsq->num_lines * image_bsq->num_samples * image_bsq->num_bands * sizeof(float));

	if (strstr(image_filename, ".bsq") == NULL) {
		printf("ERROR: Can not find file bsq: %s\n", image_filename);
		exit(1);
	}
	if ((fp = fopen(image_filename, "rb")) == NULL) {
		printf("ERROR %d. Can not open file: %s\n", errno, image_filename);
		exit(1);
	} else {
		fseek(fp, 0L, SEEK_SET);
		switch (image_bsq->data_type) {
		case 1: {
			int compno = 0;
			for(compno = 0; compno < image_bsq->num_bands; compno++) {
				for (i = 0; i < image_bsq->num_lines * image_bsq->num_samples; i++) {
					unsigned char temp;
					if (!fread(&temp, 1, 1, fp)) {
						fprintf(stderr,"Error reading raw file. End of file probably reached.\n");
						return NULL;
					}
					h_imagen[i] = ((char)temp);
				}
			}
		}
		break;
		case 2: {
/*			printf("type:2\n");
			int compno = 0;
			short value = 0;
			for(compno = 0; compno < image_bsq->num_bands; compno++) {
				for (i = 0; i < image_bsq->num_lines * image_bsq->num_samples; i++) {
					value = 0;
					unsigned char temp;
					if (!fread(&temp, 1, 1, fp)) {
						fprintf(stderr,"Error reading raw file. End of file probably reached.\n");
						return NULL;
					}
					value = temp << 8;
					if (!fread(&temp, 1, 1, fp)) {
						fprintf(stderr,"Error reading raw file. End of file probably reached.\n");
						return NULL;
					}
					value += temp;
					h_imagen[i] = ((unsigned short)value);
				}
			}*/


			type_short_int = (short *) malloc(image_bsq->num_lines * image_bsq->num_samples * image_bsq->num_bands
					* sizeof(short));

			int read_lines = fread(type_short_int, sizeof(short), (image_bsq->num_lines * image_bsq->num_samples * image_bsq->num_bands), fp);

//			printf("read:%d orginal:%d\n", read_lines, image_bsq->num_lines * image_bsq->num_samples * image_bsq->num_bands);

			int spec = 0;
			for (i = 0; i < image_bsq->num_lines * image_bsq->num_samples * image_bsq->num_bands; i++) {
				h_imagen[i] = (float) type_short_int[i];

				/*if(i == spec)
				{
					printf("%f,", h_imagen[i]);
					spec += image_bsq->num_lines * image_bsq->num_samples;
				}*/
			}
			free(type_short_int);
			break;
		}
		case 4: {
			type_float = (float *) malloc(image_bsq->num_lines * image_bsq->num_samples * image_bsq->num_bands
					* sizeof(float));
			fread(type_float, sizeof(float), (sizeof(float) * image_bsq->lines_samples * image_bsq->num_bands), fp);

			for (i = 0; i < image_bsq->num_lines * image_bsq->num_samples * image_bsq->num_bands; i++) {
				h_imagen[i] = type_float[i];
			}
			free(type_float);
			break;
		}
		case 5: {
			type_double = (double *) malloc(image_bsq->num_lines * image_bsq->num_samples * image_bsq->num_bands
					* sizeof(double));
			fread(type_double, sizeof(double), (sizeof(double) * image_bsq->lines_samples * image_bsq->num_bands), fp);

			for (i = 0; i < image_bsq->num_lines * image_bsq->num_samples * image_bsq->num_bands; i++) {
				h_imagen[i] = (float) type_double[i];
			}
			free(type_double);
			break;
		}
		}
		fclose(fp);
	}
	return h_imagen;
}

/**
 * @brief Read bsq image from file.
 *
 * @param path Path to hdr image file.
 * @param path Path to bsq image file.
 * @param _container Pointer to a malloc'ed memory for type_image.
 */
type_data *read_bsq_image(type_image *container, type_image_hyper *image_bsq, type_parameters *param)
{
	init_bsq_image(container, image_bsq, param);

	return load_image(container->in_file, image_bsq);
}
