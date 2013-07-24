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
 * @file image_hyper.c
 *
 * @author Milosz Ciznicki
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include "image_hyper.h"
#include "../print_info/print_info.h"
#include "image_types.h"
#include "../config/parameters.h"
#include "image_bsq.h"
#include "image_bip.h"
#include "image_bil.h"

/**
 * @brief Reads the header file from hyperspectral data.
 *
 * @param filename_header
 * @param image_bip
 */
void read_header(const char *filename_header, type_image_hyper *image)
{
	FILE *fp;
	char line[20];

	if (strstr(filename_header, ".hdr") == NULL) {
		printf("ERROR: Header file should have extension hdr %s\n", filename_header);
		system("PAUSE");
		exit(1);
	}

	if ((fp = fopen(filename_header, "r")) == NULL) {
		printf("ERROR %d. Can not read header file: %s \n", errno, filename_header);
		system("PAUSE");
		exit(1);
	} else {
		fseek(fp, 0L, SEEK_SET);
		while (fgets(line, 20, fp) != '\0') {
			if (strstr(line, "samples") != NULL)
				image->num_samples = atoi(strtok(strstr(line, " = "), " = "));

			if (strstr(line, "lines") != NULL)
				image->num_lines = atoi(strtok(strstr(line, " = "), " = "));

			if (strstr(line, "bands") != NULL)
				image->num_bands = atoi(strtok(strstr(line, " = "), " = "));

			if (strstr(line, "data type") != NULL)
				image->data_type = atoi(strtok(strstr(line, " = "), " = "));

		}//while
		image->lines_samples = image->num_lines * image->num_samples;
		fclose(fp);
	}//else
}

/**
 * @brief Read hyperspectral image from file.
 *
 * @param path Path to hdr image file.
 * @param path Path to bsq image file.
 * @param _container Pointer to a malloc'ed memory for type_image.
 */
int read_hyper_image(type_image **_container, type_parameters *param)
{
//	println_start(INFO);
	type_image *container = *_container;
	type_image_hyper *image = (type_image_hyper*) malloc(sizeof(type_image_hyper));
	type_data *data;

	read_header(container->in_hfile, image);

	if(container->bsq_file == 1)
		data = read_bsq_image(container, image, param);
	else if(container->bil_file == 1)
		data = read_bil_image(container, image, param);
	else if(container->bip_file == 1)
		data = read_bip_image(container, image, param);
	else {
		fprintf(stderr, "Can not load hyperspectral data set.");
		return -1;
	}

	//save_img_hyper(container, "cuprite_raw.raw");

	int band_size = container->width * container->height;

//	println_var(INFO, "Loaded %s: x:%d y:%d channels:%d depth:%d ", bsq_path, container->width, container->height,
//			container->num_components, container->depth);

	//copying data DIRECTLY as tiles to device
	int x, y, c, i;
	type_tile *tile;
	type_tile_comp *tile_comp;
	for (i = 0; i < container->num_tiles; i++) {
		tile = &(container->tile[i]);
		for (c = 0; c < container->num_components; c++) {
			tile_comp = &(tile->tile_comp[c]);
			for (y = 0; y < tile_comp->height; y++) {
				for (x = 0; x < tile_comp->width; x++) {
					tile_comp->img_data[x + y * tile_comp->width] = data[(tile->tlx + x) + (tile->tly + y)
							* tile_comp->width + c * band_size];
				}
			}
			cuda_memcpy_htd(tile_comp->img_data, tile_comp->img_data_d, tile_comp->width * tile_comp->height
					* sizeof(type_data));
			cuda_h_free(tile_comp->img_data);
		}
	}
//	free(data);
//	println_start(INFO);
	return 0;
}

void save_img_hyper(type_image *img, const char* out_file)
{
	int i, c, x, y;
	type_tile *tile;
	type_tile_comp *tile_comp;
	char** msg = (char**)calloc(1,sizeof(char*));

	float *type_float;

	type_float = (float *) malloc(img->width * img->height * img->num_components * sizeof(float));

	if(type_float == NULL) {
		asprintf(msg, "Could not allocate data!");
		perror(*msg);
	}

	int scan_width = img->width;

//	printf("num_tiles:%d num_comp:%d w:%d h:%d\n", img->num_tiles, img->num_components, img->width, img->height);
	//exact opposite procedure as in read_img()
	for (i = 0; i < img->num_tiles; i++) {
		tile = &(img->tile[i]);
		for (c = 0; c < img->num_components; c++) {
			tile_comp = &(tile->tile_comp[c]);
			cuda_h_allocate_mem((void **) &(tile_comp->img_data), tile_comp->width * tile_comp->height
					* sizeof(type_data));
			cuda_memcpy_dth(tile_comp->img_data_d, tile_comp->img_data, tile_comp->width * tile_comp->height
					* sizeof(type_data));
			for (x = 0; x < tile_comp->width; x++) {
				for (y = 0; y < tile_comp->height; y++) {
					type_float[(tile->tly + y) * tile_comp->width + (tile->tlx + x) + c * tile_comp->width * tile_comp->height]
							= tile_comp->img_data[x + y * tile_comp->width];
				}
			}
		}
	}

	/* Read output file */
	FILE *out_fp = fopen(out_file, "wb");

	printf("%d %d %d\n", img->width, img->height, img->num_components);

	if(fwrite((void*)&(img->num_components), sizeof(unsigned short int), 1, out_fp)!=1) {
		asprintf(msg, "Read component");
		perror(*msg);
		return;
	}

	if(fwrite((void*)&(img->width), sizeof(unsigned short int), 1, out_fp)!=1) {
		asprintf(msg, "Read component");
		perror(*msg);
		return;
	}

	if(fwrite((void*)&(img->height), sizeof(unsigned short int), 1, out_fp)!=1) {
		asprintf(msg, "Read component");
		perror(*msg);
		return;
	}

	if(fwrite(type_float, sizeof(float), img->width * img->height * img->num_components, out_fp)!=img->width * img->height * img->num_components) {
		asprintf(msg, "Read component");
		perror(*msg);
		return;
	}

	fclose(out_fp);
}

void write_imge(float *data, char *filename, type_image_hyper *image)
{
	FILE *fp;

	if ((fp = fopen(filename, "wb")) == NULL) {
		printf("ERROR %d. Can not write file: %s \n", errno, filename);
		system("PAUSE");
		exit(1);
	} else {
		fseek(fp, 0L, SEEK_SET);
		fwrite(data, sizeof(float), (image->num_lines * image->num_samples * image->num_bands
				* sizeof(float)), fp);
	}
	fclose(fp);
}

void write_one_band(float *data, char *filename, type_image_hyper *image)
{
	int size = image->num_lines * image->num_samples * sizeof(short int);
	short int *odata = (short int*) malloc(size);
	int i, j;

	for (j = 0; j < image->num_lines; j++) {
		for (i = 0; i < image->num_samples; i++) {
			odata[i + j * image->num_lines] = (short int) data[i + j * image->num_lines];
		}
	}

	FILE *fp;

	if ((fp = fopen(filename, "wb")) == NULL) {
		printf("ERROR %d. Can not write file: %s \n", errno, filename);
		system("PAUSE");
		exit(1);
	} else {
		fseek(fp, 0L, SEEK_SET);
		fwrite(odata, sizeof(short int), size, fp);
	}
	fclose(fp);
}
