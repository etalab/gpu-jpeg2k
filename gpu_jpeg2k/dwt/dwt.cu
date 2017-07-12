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
 * @file dwt.cu
 *
 * @author Milosz Ciznicki
 * @brief Main file.
 */

#include <stdlib.h>
#include <stdio.h>

extern "C" {
	#include "dwt.h"
	#include "../print_info/print_info.h"
	#include "../misc/memory_management.cuh"
//	#include "../preprocessing/show_image.h"
	#include "kernel.h"
}

#define DWTIRR

void fwt(type_tile *tile)
{
//	println_start(INFO);
	int i;
	type_tile_comp *tile_comp;
	type_image *img = tile->parent_img;

//	save_img(img, "enc_dwt_before.bmp");

	/* Process components from tile */
	for(i = 0; i < tile->parent_img->num_components; i++)
	{
		tile_comp = &(tile->tile_comp[i]);

//		printf("Next tile\n");

//		save_tile_comp_no_dbg(tile_comp, i);
		/* Do FWT on image data. Lossy. */
		if(img->wavelet_type)
		{
			tile_comp->img_data_d = fwt_2d(DWT97, tile_comp);
		} else /* Lossless */
		{
			tile_comp->img_data_d = fwt_2d(DWT53, tile_comp);
		}

		/*if(i == 0)
		{
			int size = tile->width * tile->height * sizeof(type_data);
			type_data *buff = (type_data*)my_malloc(size);

			cuda_memcpy_dth(tile_comp->img_data_d, buff, size);

			int x, y;
			for(y = 0; y < img->height; y++)
			{
				for(x = 0; x < img->width; x++)
				{
						printf("%6d,", (int)buff[x + y * img->width]);
				}
				printf("\n");
			}
		}*/

//		save_tile_comp_no_dbg(tile_comp, i+1);
	}
//	save_img(img, "enc_dwt_after.bmp");
//	println_end(INFO);
}

//#define TEST

#ifdef TEST
typedef struct dwt_local {
	int* mem;
	int dn;
	int sn;
	int cas;
} dwt_t;

#define S(i) a[(i)*2]
#define D(i) a[(1+(i)*2)]
#define S_(i) ((i)<0?S(0):((i)>=sn?S(sn-1):S(i)))
#define D_(i) ((i)<0?D(0):((i)>=dn?D(dn-1):D(i)))
/* new */
#define SS_(i) ((i)<0?S(0):((i)>=dn?S(dn-1):S(i)))
#define DD_(i) ((i)<0?D(0):((i)>=sn?D(sn-1):D(i)))

/* <summary>                            */
/* Inverse 5-3 wavelet transform in 1-D. */
/* </summary>                           */
static void dwt_decode_1_(int *a, int dn, int sn, int cas) {
	int i;

	if (!cas) {
		if ((dn > 0) || (sn > 1)) { /* NEW :  CASE ONE ELEMENT */
			for (i = 0; i < sn; i++) S(i) -= (D_(i - 1) + D_(i) + 2) >> 2;
			for (i = 0; i < dn; i++) D(i) += (S_(i) + S_(i + 1)) >> 1;
		}
	} else {
		if (!sn  && dn == 1)          /* NEW :  CASE ONE ELEMENT */
			S(0) /= 2;
		else {
			for (i = 0; i < sn; i++) D(i) -= (SS_(i) + SS_(i + 1) + 2) >> 2;
			for (i = 0; i < dn; i++) S(i) += (DD_(i) + DD_(i - 1)) >> 1;
		}
	}
}

/* <summary>                            */
/* Inverse 5-3 wavelet transform in 1-D. */
/* </summary>                           */
static void dwt_decode_1(dwt_t *v) {
	dwt_decode_1_(v->mem, v->dn, v->sn, v->cas);
}

/* <summary>                             */
/* Inverse lazy transform (horizontal).  */
/* </summary>                            */
static void dwt_interleave_h(dwt_t* h, int *a) {
    int *ai = a;
    int *bi = h->mem + h->cas;
    int  i	= h->sn;
    while( i-- ) {
      *bi = *(ai++);
	  bi += 2;
    }
    ai	= a + h->sn;
    bi	= h->mem + 1 - h->cas;
    i	= h->dn ;
    while( i-- ) {
      *bi = *(ai++);
	  bi += 2;
    }
}

/* <summary>                             */
/* Inverse lazy transform (vertical).    */
/* </summary>                            */
static void dwt_interleave_v(dwt_t* v, int *a, int x) {
    int *ai = a;
    int *bi = v->mem + v->cas;
    int  i = v->sn;
    while( i-- ) {
      *bi = *ai;
	  bi += 2;
	  ai += x;
    }
    ai = a + (v->sn * x);
    bi = v->mem + 1 - v->cas;
    i = v->dn ;
    while( i-- ) {
      *bi = *ai;
	  bi += 2;
	  ai += x;
    }
}

/* <summary>                            */
/* Inverse wavelet transform in 2-D.     */
/* </summary>                           */
static void dwt_decode_tile(type_tile_comp *tile_comp, int *idata) {
	int numres = tile_comp->num_dlvls + 1;
	int *sub_x, *sub_y;
	int image_size_x, image_size_y;
	dwt_t h;
	dwt_t v;
	int i;

	image_size_x = tile_comp->width / 2;
	image_size_y = tile_comp->height / 2;

	printf("%d\n", numres);

	sub_x = (int *)my_malloc((tile_comp->num_dlvls - 1) * sizeof(int));
	sub_y = (int *)my_malloc((tile_comp->num_dlvls - 1) * sizeof(int));

	for(i = 0; i < tile_comp->num_dlvls - 1; i++) {
		sub_x[i] = (image_size_x % 2 == 1) ? 1 : 0;
		sub_y[i] = (image_size_y % 2 == 1) ? 1 : 0;
		image_size_y = (int)ceil(image_size_y/2.0);
		image_size_x = (int)ceil(image_size_x/2.0);
	}

	int rw = image_size_x;	/* width of the resolution level computed */
	int rh = image_size_y;	/* height of the resolution level computed */

	int w = tile_comp->width;

	h.mem = (int*)memalign(16, tile_comp->width * sizeof(int));
	v.mem = h.mem;

	while( --numres) {
		int *tiledp = idata;
		int j;

		image_size_x = image_size_x * 2/* - sub_x[tile_comp->num_dlvls - 2 - i]*/;
		image_size_y = image_size_y * 2/* - sub_y[tile_comp->num_dlvls - 2 - i]*/;
		h.sn = rw;
		v.sn = rh;

		rw = image_size_x;
		rh = image_size_y;

		printf("rw %d rh %d h.sn %d v.sn %d\n", rw, rh, h.sn, v.sn);

		h.dn = rw - h.sn;
		h.cas = 0;

		for(j = 0; j < rh; ++j) {
			dwt_interleave_h(&h, &tiledp[j*w]);
			dwt_decode_1(&h);
			memcpy(&tiledp[j*w], h.mem, rw * sizeof(int));
		}

		v.dn = rh - v.sn;
		v.cas = 0;

		for(j = 0; j < rw; ++j){
			int k;
			dwt_interleave_v(&v, &tiledp[j], w);
			dwt_decode_1(&v);
			for(k = 0; k < rh; ++k) {
				tiledp[k * w + j] = v.mem[k];
			}
		}
	}
//	free(h.mem);
}
#endif

void iwt(type_tile *tile)
{
//	println_start(INFO);
	int i;
	type_tile_comp *tile_comp;
	type_image *img = tile->parent_img;

//	save_img(img, "dec_dwt_before.bmp");

	/* Process components from tile */
	for(i = 0; i < tile->parent_img->num_components; i++)
	{
		tile_comp = &(tile->tile_comp[i]);

		/* Do IWT on image data. Lossy. */
		if(img->wavelet_type)
		{
			tile_comp->img_data_d = iwt_2d(DWT97, tile_comp);
		} else /* Lossless */
		{
#ifdef TEST
			int *h_idata;
			type_data *h_fdata;

			h_idata = (int *)my_malloc(tile_comp->width * tile_comp->height * sizeof(int));
			h_fdata = (type_data *)my_malloc(tile_comp->width * tile_comp->height * sizeof(type_data));

			cuda_memcpy_dth(tile_comp->img_data_d, h_fdata, tile_comp->width * tile_comp->height * sizeof(type_data));

			printf("TEST\n");

			int k;

			for(k = 0; k < tile_comp->width * tile_comp->height; k++)
			{
				h_idata[k] = (int)h_fdata[k];
			}

			dwt_decode_tile(tile_comp, h_idata);

			for(k = 0; k < tile_comp->width * tile_comp->height; k++)
			{
				h_fdata[k] = (type_data)h_idata[k];
			}

			cuda_memcpy_htd(h_fdata, tile_comp->img_data_d, tile_comp->width * tile_comp->height * sizeof(type_data));
#else
			tile_comp->img_data_d = iwt_2d(DWT53, tile_comp);
#endif

		}
	}
//	save_img(img, "dec_dwt_after.bmp");
//	println_end(INFO);
}

void save_tile_comp_no_dbg(type_tile_comp *tile_comp, int i)
{
	char out_file[16];
	sprintf(out_file, "%d.bsq", i);
//	save_tile_comp(tile_comp, out_file);
}

void save_tile_comp_no_shift_dbg(type_tile_comp *tile_comp, int i)
{
	char out_file[16];
	sprintf(out_file, "%d.bsq", i);
	save_tile_comp_with_shift(tile_comp, out_file, 16);
}

void print_tile_comp(type_tile_comp *tile_comp)
{
	int i, j;
	float *odata;
	int size = tile_comp->width * tile_comp->height * sizeof(float);
	odata = (float *) my_malloc(size);
	cuda_memcpy_dth(tile_comp->img_data_d, odata, size);

	for(j = 0; j < tile_comp->height; j++)
	{
		for(i = 0; i < tile_comp->width; i++)
		{
			printf("%f,", odata[i + j * tile_comp->width]);
		}
		printf("\n");
	}
}

void fwt_dbg(type_tile *tile)
{
//	println_start(INFO);
//	start_measure();
	int i;
	type_tile_comp *tile_comp;
	type_image *img = tile->parent_img;

	int x = 0, y = 0;
	int size = tile->width * tile->height * sizeof(type_data);
	type_data *buff = (type_data*)my_malloc(size);

//	show_image(tile);
	/* Process components from tile */
	for(i = 0; i < tile->parent_img->num_components; i++)
	{
		tile_comp = &(tile->tile_comp[i]);

		cuda_memcpy_dth(tile_comp->img_data_d, buff, size);

		for(y = 0; y < tile_comp->height; y++)
		{
			for(x = 0; x < tile_comp->width; x++)
			{
				printf("%f, ", buff[x + y * tile_comp->width]);
			}
		}
		printf("\n");

		/* Do FWT on image data. Lossy. */
		if(img->wavelet_type)
		{
			tile_comp->img_data_d = fwt_2d(DWT97, tile_comp);
		} else /* Lossless */
		{
			tile_comp->img_data_d = fwt_2d(DWT53, tile_comp);
		}
	}
//	save_img(img, "dwt.bmp");
//	stop_measure(INFO);
//	show_image(tile);
//	println_end(INFO);
}
