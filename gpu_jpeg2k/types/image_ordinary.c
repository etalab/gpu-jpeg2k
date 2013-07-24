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
 * @file image.c
 * @brief This file include all the image loading/initialization/saving routines.
 *
 * @author Miłosz Ciżnicki
 * @author Jakub Misiorny <misiorny@man.poznan.pl>
 */

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <errno.h>

#include "../print_info/print_info.h"
#include "image_ordinary.h"
//#include "image_bsq.h"
#include "../preprocessing/preprocess_gpu.h"
#include "../misc/memory_management.cuh"
#include "../tier2/markers.h"
//#include "show_image.h"

#include "../config/parameters.h"

#include <FreeImage.h>
//#include "dbg_image.h"

//#define READ_TIME

/**
 * @brief Initializes image container
 *
 * @param dib Bitmap opened by FreeImage
 * @param container malloc'ed type_image to be initialized.
 */
void init_image(FIBITMAP* dib, type_image *container, type_parameters *param)
{
	//	println_start(INFO);
	container->height = FreeImage_GetHeight(dib);
	container->width = FreeImage_GetWidth(dib);
	container->depth = FreeImage_GetBPP(dib);
	container->num_components = get_num_comp(dib);
	container->num_range_bits = container->depth / container->num_components;

	/* TODO: check if origin data was signed? */
	container->sign = get_img_type(dib);
	container->num_dlvls = param->param_tile_comp_dlvls;

	set_coding_parameters(container, param);
	init_tiles(&container, param);

//	println_end(INFO);
}

uint8_t get_num_comp(FIBITMAP* dib)
{
	FREE_IMAGE_COLOR_TYPE color_type = FreeImage_GetColorType(dib);
	switch(color_type)
	{
	case FIC_MINISWHITE: /*printf("FIC_MINISWHITE\n");*/ return 1U;
	case FIC_MINISBLACK: /*printf("FIC_MINISBLACK\n");*/ return 1U;
	case FIC_RGB: /*printf("RGB\n");*/ return 3U;
	default: printf("Unsupported image type!\n"); exit(0);
	/*case FIC_PALETTE: printf("FIC_PALETTE\n"); break;
	case FIC_RGBALPHA: printf("FIC_RGBALPHA\n"); break;
	case FIC_CMYK: printf("FIC_CMYK\n"); break;*/
	}
}

uint8_t get_img_type(FIBITMAP* dib)
{
	FREE_IMAGE_TYPE image_type = FreeImage_GetImageType(dib);

	switch(image_type)
	{
	case FIT_BITMAP:
	case FIT_UINT16:
	case FIT_UINT32:
	case FIT_RGB16:
	case FIT_RGBA16: return UNSIGNED;
	case FIT_INT16:
	case FIT_INT32: return SIGNED;
	}
	return 2U;
}

/**
FreeImage error handler
@param fif Format / Plugin responsible for the error
@param message Error message
*/
void FreeImageErrorHandler(FREE_IMAGE_FORMAT fif, const char *message) {
   printf("\n*** ");
   if(fif != FIF_UNKNOWN) {
     printf("%s Format\n", FreeImage_GetFormatFromFIF(fif));
   }
   printf("%s", message);
   printf(" ***\n");
}

/*void convert(const char *filename)
{
	int ret;
	char cmdline[1048];

	char temp_file[32];
	tmpnam(temp_file); //get temp filename

	//printf("temp file is: '%s'\n", temp_file);

	//Create Windows bitmap 3.0 with 24bpp
	sprintf(cmdline, "%s \"%s\" -type truecolor \"BMP3:%s\"", "convert", filename, temp_file);
	//printf("cmdline = '%s'\n", cmdline);
	ret = system(cmdline);
	//system("pause");
	if(ret != 0) {
		printf("Could not convert '%s' to temp BMP '%s' !\n", filename, temp_file);
		printf("convert returned error %d\n", ret);
		printf("cmdline was: '%s'\n", cmdline);
		free(temp_file);
		exit(0);
	}
	strcpy(filename, temp_file);
}*/


/**
 * @brief Read image from file.
 *
 * @param path Path to image file.
 * @param _container Pointer to a malloc'ed memory for type_image.
 */
long int read_ordinary_image(type_image **_container, type_parameters *param)
{
	//	println_start(INFO);
#ifdef READ_TIME
	long int start_load;
	start_load = start_measure();
#endif

	type_image *container = *_container;
	FREE_IMAGE_FORMAT formato = FreeImage_GetFileType(container->in_file, 0);
	FIBITMAP* dib = FreeImage_Load(formato, container->in_file, 0);
	FREE_IMAGE_TYPE image_type = FreeImage_GetImageType(dib);
	long int copy_time = 0;

#ifdef READ_TIME
	cudaThreadSynchronize();
	printf("Load img:%ld\n", stop_measure(start_load));
#endif

	FreeImage_SetOutputMessage(FreeImageErrorHandler);

	if(image_type == FIT_BITMAP)
	{
//		printf("BITMAP\n");
#ifdef READ_TIME
		long int start_convert;
		start_convert = start_measure();
#endif

		dib = FreeImage_ConvertTo24Bits(dib);

#ifdef READ_TIME
		cudaThreadSynchronize();
		printf("Convert img:%ld\n", stop_measure(start_convert));
#endif

		if(FreeImage_HasPixels(dib) == FALSE)
		{
			printf("Do not have pixel data!\n");
			exit(0);
		}

#ifdef READ_TIME
		long int start_read;
		start_read = start_measure();
#endif

		init_image(dib, container, param);

#ifdef READ_TIME
		cudaThreadSynchronize();
		printf("Init img:%ld\n", stop_measure(start_read));
#endif

//		println_var(INFO, "Loaded %s: x:%d y:%d channels:%d depth:%d ",
//				path, container->width, container->height, container->num_components, container->depth);

#ifdef READ_TIME
		long int start_raw;
		start_raw = start_measure();
#endif

		int scan_width = FreeImage_GetPitch(dib);
		int mem_size = container->height * scan_width;

		BYTE *bits = malloc(mem_size * sizeof(BYTE));

		// convert the bitmap to raw bits (top-left pixel first)
		FreeImage_ConvertToRawBits(bits, dib, scan_width, FreeImage_GetBPP(dib), FI_RGBA_RED_MASK/*FI_RGBA_BLUE_MASK*/, FI_RGBA_GREEN_MASK, /*FI_RGBA_RED_MASK*/FI_RGBA_BLUE_MASK, TRUE);
//		FreeImage_ConvertToRawBits(bits, dib, scan_width, FreeImage_GetBPP(dib), FI_RGBA_GREEN_MASK/*FI_RGBA_BLUE_MASK*/, FI_RGBA_BLUE_MASK, /*FI_RGBA_RED_MASK*/FI_RGBA_RED_MASK, TRUE);

		FreeImage_Unload(dib);
#ifdef READ_TIME
		cudaThreadSynchronize();
		printf("Raw bits img:%ld\n", stop_measure(start_raw));
#endif

		//FreeImage_Unload(dib);

		long int start_copy;
		start_copy = start_measure();

		//copying data DIRECTLY as tiles to device
		int x, y, c, i;
		type_tile *tile;
		type_tile_comp *tile_comp;
		for(i = 0; i < container->num_tiles; i++) {
			tile = &(container->tile[i]);
			for(c = 0; c < container->num_components; c++) {
				tile_comp = &(tile->tile_comp[c]);
				for(y = 0; y < tile_comp->height; y++) {
					for(x = 0; x < tile_comp->width; x++) {
						tile_comp->img_data[x + y*tile_comp->width] =
						(type_data)bits[(tile->tly + y) * scan_width + (tile->tlx + x) * container->num_components + c];
//						if(c == 0)
//							printf("%6d,", bits[(tile->tly + y) * scan_width + (tile->tlx + x) * container->num_components + c] - 128);
					}
//					if(c == 0)
//						printf("\n");
				}
				cuda_memcpy_htd(tile_comp->img_data, tile_comp->img_data_d, tile_comp->width * tile_comp->height * sizeof(type_data));
				cuda_h_free(tile_comp->img_data);
//				free(tile_comp->img_data);
			}
		}

//		cudaThreadSynchronize();
		copy_time = stop_measure(start_copy);

/*		char buff[128];

		sprintf(buff, "%s.raw\0", path);

		save_raw(container, buff);*/

#ifdef READ_TIME
		printf("Copy img:%ld\n", copy_time);
#endif

		//free(bits);
	} else if(image_type == FIT_RGB16 || image_type == FIT_UINT16)
	{
//		printf("RGB16 or UINT16\n");
		init_image(dib, container, param);

		int scan_width = FreeImage_GetPitch(dib)/sizeof(unsigned short);

		println_var(INFO, "Loaded %s: x:%d y:%d channels:%d depth:%d ",
		container->in_file, container->width, container->height, container->num_components, container->depth);

		//copying data DIRECTLY as tiles to device
		int x, y, c, i;
		type_tile *tile;
		type_tile_comp *tile_comp;
		for(i = 0; i < container->num_tiles; i++) {
			tile = &(container->tile[i]);
			for(c = 0; c < container->num_components; c++) {
				tile_comp = &(tile->tile_comp[c]);
				for(y = 0; y < tile_comp->height; y++) {
					for(x = 0; x < tile_comp->width; x++) {
						tile_comp->img_data[x + (tile_comp->height - 1 - y)*tile_comp->width] =
						(type_data)((unsigned short *)FreeImage_GetBits(dib))[(tile->tly + y) * scan_width + (tile->tlx + x) * container->num_components + c];
					}
				}
				cuda_memcpy_htd(tile_comp->img_data, tile_comp->img_data_d, tile_comp->width * tile_comp->height * sizeof(type_data));
				cuda_h_free(tile_comp->img_data);
			}
		}
		FreeImage_Unload(dib);
	} else
	{
		init_image(dib, container, param);
		dib = FreeImage_ConvertToType(dib, FIT_DOUBLE, TRUE);

		if(dib == NULL)
		{
			printf("Conversion not allowed!\n");
			exit(0);
		}

		int scan_width = FreeImage_GetPitch(dib)/sizeof(double);

		if(FreeImage_HasPixels(dib) == FALSE)
		{
			printf("Do not have pixel data!\n");
			exit(0);
		}
		println_var(INFO, "Loaded %s: x:%d y:%d channels:%d depth:%d ",
		container->in_file, container->width, container->height, container->num_components, container->depth);

		//copying data DIRECTLY as tiles to device
		int x, y, c, i;
		type_tile *tile;
		type_tile_comp *tile_comp;
		for(i = 0; i < container->num_tiles; i++) {
			tile = &(container->tile[i]);
			for(c = 0; c < container->num_components; c++) {
				tile_comp = &(tile->tile_comp[c]);
				for(y = 0; y < tile_comp->height; y++) {
					for(x = 0; x < tile_comp->width; x++) {
						tile_comp->img_data[x + (tile_comp->height - 1 - y)*tile_comp->width] =
						(type_data)((double *)FreeImage_GetBits(dib))[(tile->tly + y) * scan_width + (tile->tlx + x) * container->num_components + c];
					}
				}
				cuda_memcpy_htd(tile_comp->img_data, tile_comp->img_data_d, tile_comp->width * tile_comp->height * sizeof(type_data));
				cuda_h_free(tile_comp->img_data);
			}
		}
		FreeImage_Unload(dib);
	}

	return 0;
}

void save_tile_comp(type_tile_comp *tile_comp, char *filename)
{
	float *image;
	int size = tile_comp->width * tile_comp->height;
	image = (float *) malloc(size * sizeof(float));
	cuda_memcpy_dth(tile_comp->img_data_d, image, size * sizeof(float));

	short int *odata = (short int*) malloc(size * sizeof(short int));
	int i, j;

	for (j = 0; j < tile_comp->height; j++) {
		for (i = 0; i < tile_comp->width; i++) {
			odata[i + j * tile_comp->width] = (short int) image[i + j * tile_comp->width];
		}
	}

	FILE *fp;

	if ((fp = fopen(filename, "wb")) == NULL) {
		printf("ERROR %d. Can not write file: %s \n", errno, filename);
		system("PAUSE");
		exit(1);
	} else {
		fseek(fp, 0L, SEEK_SET);
		fwrite(odata, sizeof(short int), size * sizeof(short int), fp);
	}
	fclose(fp);
	free(odata);
	free(image);
}

void save_tile_comp_with_shift(type_tile_comp *tile_comp, char *filename, int shift)
{
	float *odata;
	int size = tile_comp->width * tile_comp->height * sizeof(float);
	odata = (float *) malloc(size);
	cuda_memcpy_dth(tile_comp->img_data_d, odata, size);
	int i, j;

	for (j = 0; j < tile_comp->height; j++) {
		for (i = 0; i < tile_comp->width; i++) {
			odata[i + j * tile_comp->width] += 1 << (shift - 1);
			printf("%.1f,", odata[i + j * tile_comp->width]);
		}
		printf("\n");
	}

	FILE *fp;
	if ((fp = fopen(filename, "wb")) == NULL) {
		printf("ERROR %d. Can not open file: %s \n", errno, filename);
		system("PAUSE");
		exit(1);
	} else {
		fseek(fp, 0L, SEEK_SET);
		fwrite(odata, sizeof(float), size, fp);
	}
	fclose(fp);
	free(odata);
}

int save_img_ord(type_image *img, const char *filename)
{
	println_start(INFO);
	int i, c, x, y;
	type_tile *tile;
	type_tile_comp *tile_comp;

	int scan_width = img->width * (img->num_components);
	BYTE *bits = (BYTE*) malloc(img->height * scan_width * sizeof(BYTE));

	//	printf("for\n");
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
					bits[(tile->tly + y) * scan_width + (tile->tlx + x) * img->num_components + c]
							= (BYTE) tile_comp->img_data[x + y * tile_comp->width];

					//					if(tile_comp->img_data[x + y * tile_comp->width]  > (BYTE) tile_comp->img_data[x + y * tile_comp->width]) {
					//						printf("%f = %i\n", tile_comp->img_data[x + y * tile_comp->width], bits[(tile->tly + y) * scan_width + (tile->tlx + x) * img->num_components + c]);
					//					}
				}
			}
		}
	}

	// convert the bitmap to raw bits (top-left pixel first)
	FIBITMAP *dst = FreeImage_ConvertFromRawBits(bits, img->width, img->height, scan_width, 24, FI_RGBA_RED_MASK,
			FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);

	if (FreeImage_Save(FIF_BMP, dst, filename, 0)) {
		println_var(INFO, "saved: %s", filename);
	} else {
		println_var(INFO, "saving FAILED: %s", filename);
	}

	FreeImage_Unload(dst);
	free(bits);

	return 0;

}

int save_img_grayscale(type_image *img, char *filename)
{
	println_start(INFO);
	int i, c, x, y;
	type_tile *tile;
	type_tile_comp *tile_comp;

	tile = &(img->tile[0]);
	tile_comp = &(tile->tile_comp[0]);
	cuda_h_allocate_mem((void **) &(tile_comp->img_data), tile_comp->width * tile_comp->height
			* sizeof(type_data));
	cuda_memcpy_dth(tile_comp->img_data_d, tile_comp->img_data, tile_comp->width * tile_comp->height
			* sizeof(type_data));

	int scan_width = img->width * (img->num_components);
	BYTE *bits = (BYTE*) malloc(img->height * scan_width * sizeof(BYTE));

	//	printf("for\n");
	//exact opposite procedure as in read_img()
	for (i = 0; i < img->num_tiles; i++) {
		for (c = 0; c < img->num_components; c++) {
			for (x = 0; x < tile_comp->width; x++) {
				for (y = 0; y < tile_comp->height; y++) {
					bits[(tile->tly + y) * scan_width + (tile->tlx + x) * img->num_components + c]
							= (BYTE) tile_comp->img_data[x + y * tile_comp->width];

					//					if(tile_comp->img_data[x + y * tile_comp->width]  > (BYTE) tile_comp->img_data[x + y * tile_comp->width]) {
					//						printf("%f = %i\n", tile_comp->img_data[x + y * tile_comp->width], bits[(tile->tly + y) * scan_width + (tile->tlx + x) * img->num_components + c]);
					//					}
				}
			}
		}
	}

	// convert the bitmap to raw bits (top-left pixel first)
	FIBITMAP *dst = FreeImage_ConvertFromRawBits(bits, img->width, img->height, scan_width, 24, FI_RGBA_RED_MASK,
			FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);

	if (FreeImage_Save(FIF_BMP, dst, filename, 0)) {
		println_var(INFO, "saved: %s", filename);
	} else {
		println_var(INFO, "saving FAILED: %s", filename);
	}

	FreeImage_Unload(dst);
	free(bits);

	return 0;

}
