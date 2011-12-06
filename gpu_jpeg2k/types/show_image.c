/*
 * @file show_image.c
 *
 * @author Milosz Ciznicki 
 * @date 03-12-2010
 */

//#include <cv.h>
//#include <highgui.h>
#include <stdio.h>

#include "image_types.h"
#include "allocate_memory.cuh"

void show_image(type_tile *tile)
{
	int num_components = tile->parent_img->num_components;
	// Allocate a float image
	IplImage* img = cvCreateImage(cvSize(tile->width, tile->height), IPL_DEPTH_32F, num_components);
	float *odata = (float *)img->imageData;
	int step = img->widthStep / sizeof(float);
	int i, j, k;
	float *idata = NULL;

	type_tile_comp *tile_comp;

	cuda_h_allocate_mem((void **)&idata, tile->width * tile->height * sizeof(float));

	for(k = 0; k < num_components; k++)
	{
		tile_comp = &(tile->tile_comp[k]);
		cuda_memcpy_dth(tile_comp->img_data_d, idata, tile_comp->width * tile_comp->height * sizeof(float));
		for(i = 0; i < tile_comp->width; i++)
		{
			for(j = 0; j < tile_comp->height; j++)
			{
				odata[k + i * num_components + j * step] = idata[i + j * tile_comp->width];
			}
		}
	}


	cvNamedWindow( "ImageView", CV_WINDOW_NORMAL);

	if(tile->width > 1920 || tile->height > 1200)
	{
		int ratio = 1200 * ((float)tile->width / (float)tile->height);
		cvResizeWindow("ImageView", ratio, 1200);
	}
	cvShowImage( "ImageView", img );
	cvWaitKey(0); // very important, contains event processing loop inside
	cvDestroyWindow( "ImageView" );
	cvReleaseImage( &img );
}

void show_imagef(type_data *data, int width, int height)
{
	// Allocate a float image
	IplImage* img = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);
	float *odata = (float *)img->imageData;
	int step = img->widthStep / sizeof(float);
	int i, j;
	float *idata = NULL;

	cuda_h_allocate_mem((void **)&idata, width * height * sizeof(float));
	cuda_memcpy_dth(data, idata, width * height * sizeof(float));

	for(i = 0; i < width; i++)
	{
		for(j = 0; j < height; j++)
		{
			odata[i + j * step] = idata[i + j * width];
		}
	}

	cvNamedWindow( "Image view", 1 );
	cvShowImage( "Image view", img );
	cvWaitKey(0); // very important, contains event processing loop inside
	cvDestroyWindow( "Image view" );
	cvReleaseImage( &img );
}

void show_imagec(char *data, int width, int height)
{
	// Allocate a float image
	IplImage* img = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
	char *odata = (char *)img->imageData;
	int step = img->widthStep / sizeof(char);
	int i, j;

	for(i = 0; i < width; i++)
	{
		for(j = 0; j < height; j++)
		{
			odata[i + j * step] = data[i + j * width];
		}
	}

	cvNamedWindow( "Image view", 1 );
	cvShowImage( "Image view", img );
	cvWaitKey(0); // very important, contains event processing loop inside
	cvDestroyWindow( "Image view" );
	cvReleaseImage( &img );
}
