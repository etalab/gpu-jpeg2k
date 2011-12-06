/*
 * @file dbg_fwt_1d.cu
 *
 * @author Milosz Ciznicki 
 * @date 06-06-2011
 */
extern "C" {
#include "dbg_fwt_1d.h"
#include "../dwt/fwt_1d.h"
#include "../misc/memory_management.cuh"
}

#define NUM_COMP 12
#define LEVEL 1

void dbg_fwt_1d()
{
	type_image *img = (type_image *)malloc(sizeof(type_image));

	img->num_components = NUM_COMP;
	img->width = 1;
	img->height = 1;

	type_tile *tile = (type_tile *) malloc(sizeof(type_tile));
	type_tile_comp *tile_comp = (type_tile_comp *)  malloc(sizeof(type_tile_comp) * img->num_components);

	float input[NUM_COMP] = {3, 44, 6, 7, 8, 9, 4, 45, 7, 8, 9, 10};

	int i;
	for(i = 0; i < img->num_components; ++i)
	{
		tile_comp[i].width = img->width;
		tile_comp[i].height = img->height;
		cuda_d_allocate_mem((void **)&tile_comp[i].img_data_d, img->width * img->height * sizeof(type_data));
		cuda_memcpy_htd(&input[i], tile_comp[i].img_data_d, img->width * img->height * sizeof(type_data));
	}

	tile->tile_comp = tile_comp;
	img->tile = tile;

	fwt_1d(img, LEVEL);

	type_data **output = (type_data **)malloc(img->num_components * sizeof(type_data *));

	for(i = 0; i < img->num_components; ++i)
	{
		output[i] = (type_data *) malloc(img->width * img->height * sizeof(type_data));

		cuda_memcpy_dth(tile_comp[i].img_data_d, output[i], img->width * img->height * sizeof(type_data));
	}

	int j;
	for(j = 0; j < img->width * img->height; ++j)
	{
		for(i = 0; i < img->num_components; ++i)
		{
			printf("%f,", output[i][j]);
		}
		printf("\n");
	}
}
