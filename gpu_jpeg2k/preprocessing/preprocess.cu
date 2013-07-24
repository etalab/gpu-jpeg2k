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
 * @file: preprocess.cu
 *
 * @author: Jakub Misiorny <misiorny@man.poznan.pl>
 */

#include <stdlib.h>
#include <stdio.h>

#include <sys/time.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime_api.h>


#include "preprocess_gpu.cuh"
#include "allocate_memory.h"
#include "cuda_errors.h"

//extern

typedef unsigned char uchar;

#include "image.h"

/**
 * Returns time in mili-seconds. 1 ms = 0.001 s.
 */
inline long double period(struct timeval a, struct timeval b) {
	return 1000 * (b.tv_sec - a.tv_sec) 	+ (b.tv_usec - a.tv_usec)/1000;
}

/*
type_image *dc_level_shift(type_image *array) {
	int i = 0;
	while(i < array->x_size * array->y_size * array->channels) {
	//	printf("(%f) = ", array->data[i]);
		array->data[i++] -= 128.0;
	//	printf("%f\n", array->data[i-1]);
	}
	return array;
}
*/

/*

type_image rct_cpu(type_image *img) {
	float b,g,r;
	float y,u,v;
	int i,j;

	//printf("[rct]processing: x: %i, y:%i c:%i\n", img->x_size, img->y_size, img->channels);
	type_image *yuv = (type_image *)malloc(sizeof(type_image));
	yuv->float_data = (float *)malloc(img->x_size * img->y_size * img->channels * sizeof(float));
	
	yuv->x_size = img->x_size;
	yuv->y_size = img->y_size;
	yuv->channels = img->channels;
	yuv->wstep = img->wstep;
	
	for(i=0;i<img->y_size;i++) {
			for(j=0; j < img->x_size ; j++) {
				b = (float)img->uchar_data[i*(img->x_size*img->channels) + j*img->channels + 0];
				g = (float)img->uchar_data[i*(img->x_size*img->channels) + j*img->channels + 1];
				r = (float)img->uchar_data[i*(img->x_size*img->channels) + j*img->channels + 2];

				y = floor( (r + 2*g + b)/4) ;
				u = b - g;
				v = r - g;

				yuv->float_data[i*(img->x_size*img->channels) + j*img->channels + 0] = y;
				yuv->float_data[i*(img->x_size*img->channels) + j*img->channels + 1] = u;
				yuv->float_data[i*(img->x_size*img->channels) + j*img->channels + 2] = v;
			}
	}
	return yuv;
}

type_image *tcr_cpu(type_image *img) {
	uchar b,g,r;
	float y,u,v;
	int i,j;

	//printf("[tcr]processing: x: %i, y:%i c:%i\n", img->x_size, img->y_size, img->channels);
	type_image *rgb = (type_image *)malloc(sizeof(type_image));
	rgb->uchar_data = (uchar *)malloc(img->x_size * img->y_size * img->channels * sizeof(uchar));
	
	rgb->x_size = img->x_size;
	rgb->y_size = img->y_size;
	rgb->channels = img->channels;
	rgb->wstep = img->wstep;
	rgb->float_data = NULL;

	for(i=0;i<img->y_size;i++) {
		for(j=0; j < img->x_size ; j++) {
			y = img->float_data[i*(img->x_size*img->channels) + j*img->channels + 0];
			u = img->float_data[i*(img->x_size*img->channels) + j*img->channels + 1];
			v = img->float_data[i*(img->x_size*img->channels) + j*img->channels + 2];


			g = (uchar)(y - floor((u + v) / 4) );
			r = (uchar)(v + g);
			b = (uchar)(u + g);

			rgb->uchar_data[i*(img->x_size*img->channels) + j*img->channels + 0] = b;
			rgb->uchar_data[i*(img->x_size*img->channels) + j*img->channels + 1] = g;
			rgb->uchar_data[i*(img->x_size*img->channels) + j*img->channels + 2] = r;
		}
	}
	return rgb;
}

*/

int main(int argc, char **argv) {
	struct timeval start, end;
	
	printf("%s %s %s\n", argv[0], argv[1], argv[2]);

	type_image *img = 0;
	//cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostAlloc(&img, sizeof(type_image), cudaHostAllocPortable);
	char *filename = argv[1];
	read_ordinary_image(&img, filename);
	printf("[main]: loaded\n");
	checkCUDAError("read_img");

	printf("[main]: x: %i, y: %i, channels: %i, \n", img->x_size, img->y_size, img->channels);

//	printf("dc level\n");
//	img = dc_level_shift(img);

/*
	save_img(img, "dc.png");
	gettimeofday(&start, NULL);
		type_image *yuv = rct_cpu(img);
	gettimeofday(&end, NULL);
	printf("\tRCT CPU: %.2Lf [ms]\n", period(start, end));
	save_img(yuv, "yuv.png");

	gettimeofday(&start, NULL);
		type_image *rgb = tcr_cpu(yuv);
	gettimeofday(&end, NULL);
	printf("\tRCT Inverse CPU: %.2Lf [ms]\n", period(start, end));
	save_img(rgb, "rgb.png");

	if(yuv->float_data != NULL)
		free(yuv->float_data);
	free(yuv);

*/
/*
	printf("\tRCT GPU: \n");
	type_image *yuv_gpu = color_trans_gpu(img, RCT);
	save_img(yuv_gpu, "yuv_gpu.png");

	printf("\tTCR GPU: \n");
	type_image *rgb_gpu = color_trans_gpu(yuv_gpu, TCR);
	save_img(rgb_gpu, "rgb_gpu.png");
	
	printf("\tICT GPU:\n");
	type_image *ict_gpu = color_trans_gpu(img, ICT);
	save_img(ict_gpu, "ict_gpu.png");

	printf("\tTCI GPU:\n");
	type_image *tci_gpu = color_trans_gpu(ict_gpu, TCI);
	save_img(tci_gpu, "tci_gpu.png");

	cudaFreeHost(img);
	return 0;
	*/
}
