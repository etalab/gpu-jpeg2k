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
 * @file dbg_fiwt_1d.cu
 *
 * @author Milosz Ciznicki
 */

extern "C" {
#include "dbg_fiwt_1d.h"
#include "../dwt/fwt_1d.h"
#include "../dwt/iwt_1d.h"
#include "../misc/memory_management.cuh"
}

#define NUM_COMP 188
#define LEVEL 4

void dbg_fiwt_1d()
{
	int printf_buff = 10 * 1024 * 1024;

	cuda_set_printf_limit(printf_buff);

	type_image *img = (type_image *) my_malloc(sizeof(type_image));

	img->num_components = NUM_COMP;
	img->width = 1;
	img->height = 1;

	type_tile *tile = (type_tile *) my_malloc(sizeof(type_tile));
	type_tile_comp *tile_comp = (type_tile_comp *) my_malloc(sizeof(type_tile_comp) * img->num_components);

//	float input[NUM_COMP] = {3, 44, 6, 7, 8, 9, 4, 45, 7, 8, 9};
	float input[NUM_COMP] = { 726.000000, 905.000000, 951.000000, 1040.000000, 1095.000000, 1138.000000, 1161.000000,
			1181.000000, 1184.000000, 1209.000000, 1247.000000, 1291.000000, 1357.000000, 1418.000000, 1490.000000,
			1582.000000, 1678.000000, 1749.000000, 1827.000000, 1887.000000, 1929.000000, 1976.000000, 1999.000000,
			2021.000000, 2053.000000, 2080.000000, 2111.000000, 2149.000000, 2184.000000, 2105.000000, 2128.000000,
			2179.000000, 2226.000000, 2264.000000, 2316.000000, 2361.000000, 2405.000000, 2442.000000, 2460.000000,
			2484.000000, 2488.000000, 2468.000000, 2480.000000, 2453.000000, 2443.000000, 2445.000000, 2416.000000,
			2411.000000, 2390.000000, 2374.000000, 2375.000000, 2360.000000, 2383.000000, 2388.000000, 2377.000000,
			2373.000000, 2387.000000, 2460.000000, 2444.000000, 2469.000000, 2494.000000, 2510.000000, 2527.000000,
			2546.000000, 2577.000000, 2595.000000, 2622.000000, 2665.000000, 2706.000000, 2730.000000, 2771.000000,
			2823.000000, 2852.000000, 2885.000000, 2911.000000, 2962.000000, 3021.000000, 3056.000000, 3066.000000,
			3168.000000, 3226.000000, 3216.000000, 3260.000000, 3290.000000, 3332.000000, 3359.000000, 3404.000000,
			3440.000000, 3489.000000, 3521.000000, 3567.000000, 3601.000000, 3598.000000, 3611.000000, 3686.000000,
			3690.000000, 3712.000000, 3726.000000, 3745.000000, 3746.000000, 3781.000000, 3659.000000, 3678.000000,
			3693.000000, 3698.000000, 3757.000000, 3780.000000, 3744.000000, 3807.000000, 3823.000000, 3838.000000,
			3851.000000, 3863.000000, 3895.000000, 3919.000000, 3928.000000, 3923.000000, 3952.000000, 3978.000000,
			3984.000000, 4011.000000, 4001.000000, 4020.000000, 4035.000000, 4028.000000, 4004.000000, 4044.000000,
			4045.000000, 4022.000000, 4008.000000, 4027.000000, 4034.000000, 4046.000000, 4035.000000, 4060.000000,
			4103.000000, 3337.000000, 3372.000000, 3477.000000, 3744.000000, 3717.000000, 3650.000000, 3612.000000,
			3668.000000, 3605.000000, 3656.000000, 3655.000000, 3596.000000, 3530.000000, 3568.000000, 3563.000000,
			3546.000000, 3556.000000, 3503.000000, 3445.000000, 3351.000000, 3217.000000, 3039.000000, 2875.000000,
			2928.000000, 3062.000000, 3223.000000, 3255.000000, 3249.000000, 3275.000000, 3309.000000, 3249.000000,
			3221.000000, 3180.000000, 3014.000000, 3092.000000, 2941.000000, 2846.000000, 2783.000000, 2796.000000,
			2772.000000, 2857.000000, 2784.000000, 2736.000000, 2621.000000, 2500.000000, 2496.000000, 2481.000000,
			2282.000000, 2376.000000, 2427.000000, 2271.000000, 2192.000000 };

/*	float input[NUM_COMP] = { 726.000000, 905.000000, 951.000000, 1040.000000, 1095.000000, 1138.000000, 1161.000000,
			1181.000000, 1184.000000, 1209.000000, 1247.000000};*/

	int i, j;

	/*	srand(time(NULL));

	 for(i = 0; i < NUM_COMP; ++i)
	 {
	 input = rand();
	 }*/

	for (j = 0; j < img->width * img->height; ++j) {
		for (i = 0; i < img->num_components; ++i) {
			printf("%f,", input[i]);
		}
		printf("\n");
	}

	for (i = 0; i < img->num_components; ++i) {
		tile_comp[i].width = img->width;
		tile_comp[i].height = img->height;
		cuda_d_allocate_mem((void **) &tile_comp[i].img_data_d, img->width * img->height * sizeof(type_data));
		cuda_memcpy_htd(&input[i], tile_comp[i].img_data_d, img->width * img->height * sizeof(type_data));
	}

	tile->tile_comp = tile_comp;
	img->tile = tile;

	fwt_1d(img, LEVEL);

	type_data **output = (type_data **) my_malloc(img->num_components * sizeof(type_data *));

	for (i = 0; i < img->num_components; ++i) {
		output[i] = (type_data *) my_malloc(img->width * img->height * sizeof(type_data));

		cuda_memcpy_dth(tile_comp[i].img_data_d, output[i], img->width * img->height * sizeof(type_data));
	}

	for (j = 0; j < img->width * img->height; ++j) {
		for (i = 0; i < img->num_components; ++i) {
			printf("%f,", output[i][j]);
		}
		printf("\n");
	}

	iwt_1d(img, LEVEL);

	for (i = 0; i < img->num_components; ++i) {
		cuda_memcpy_dth(tile_comp[i].img_data_d, output[i], img->width * img->height * sizeof(type_data));
	}

	for (j = 0; j < img->width * img->height; ++j) {
		for (i = 0; i < img->num_components; ++i) {
			printf("%f,", output[i][j]);
		}
		printf("\n");
	}
}
