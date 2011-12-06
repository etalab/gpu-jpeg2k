/*
 * mct.c
 *
 *  Created on: Dec 1, 2011
 *      Author: miloszc
 */

#include "mct.h"
#include "../types/image_types.h"
#include "../klt/klt.h"

void mct(type_image *img, type_parameters *param) {
	/* Multicomponent transform and DC */
	if(img->use_mct == 1)
	{
		//lossless coder
		if(img->wavelet_type == 0) {
			//printf("Lossless\n");
			color_coder_lossless(img);
		}
		else //lossy coder
		{
//			printf("Lossy\n");
			color_coder_lossy(img);
		}
	} else if (img->use_part2_mct == 1) {
		if(img->mct_compression_method == 0)
		{
			img->mct_data = (type_multiple_component_transformations*)calloc(1, sizeof(type_multiple_component_transformations));
			encode_klt(param, img);
		} else if((img->mct_compression_method == 2))
		{
//			fwt_1d(img, 4);
		}

	} else {
		if(img->sign == UNSIGNED)
		{
			//printf("unsigned\n");
			fdc_level_shifting(img);
		}
	}
}
