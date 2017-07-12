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
#include "image_bsq.h"
#include "../preprocessing/preprocess_gpu.h"
#include "../misc/memory_management.cuh"
#include "../tier2/markers.h"
#include "image_hyper.h"
#include "image_ordinary.h"
//#include "show_image.h"

#include "../config/parameters.h"

#include <FreeImage.h>
//#include "dbg_image.h"

//#define READ_TIME


/**
 * @brief Debugging functions which prints raw image data into the log. Use only with really small images. It produces ridiculous amounts of data for bigger images.
 */
/*void print_image(type_image *img)
 {
 //	println_start(INFO);
 uint16_t h = img->height, w = img->width;
 uint64_t mem_size = img->height * img->width * img->num_components * sizeof(type_data);
 type_data *data;

 cuda_h_allocate_mem((void **) &data, mem_size);
 cuda_memcpy_dth(img->img_data_d, data, mem_size);

 int i, j, c, x, y;
 int channels = img->num_components;
 int step = 8;
 for (c = 0; c < img->num_components; c++) {
 for (y = 0; y < h; y++) {
 for (x = 0; x < w; x++) {
 printf("[x: %i y: %i c: %i](%i): %f\n", x, y, c, c * img->width * img->height + y * img->width + x,
 data[c * h * w + y * w + x]);
 }
 }
 printf("\n");
 }

 cuda_h_free(data);
 //	println_end(INFO);
 }*/

/**
 * @brief Initializes codeblocks in the specific subband.
 *
 * @param sb Subband to initialize.
 */
void init_codeblocks(type_subband *sb) {
	//	println_start(INFO);
	uint32_t i;
	type_codeblock *cblk;
	type_tile_comp *tile_comp;

	sb->cblks = (type_codeblock *) malloc(sb->num_cblks * sizeof(type_codeblock));

	tile_comp = sb->parent_res_lvl->parent_tile_comp;

	//	println_var(INFO, "sb:tlx:%d tly:%d brx:%d bry:%d w:%d h:%d num_xcblks:%d num_ycblks:%d num_cblks:%d", sb->tlx, sb->tly, sb->brx, sb->bry, sb->width, sb->height, sb->num_xcblks, sb->num_ycblks, sb->num_cblks);

	for (i = 0; i < sb->num_cblks; i++) {
		cblk = &(sb->cblks[i]);
		cblk->cblk_no = i;

		cblk->no_x = cblk->cblk_no % sb->num_xcblks;
		cblk->no_y = cblk->cblk_no / sb->num_xcblks;

		cblk->tlx = cblk->no_x * tile_comp->cblk_w;
		cblk->tly = cblk->no_y * tile_comp->cblk_h;

		cblk->brx = (cblk->no_x + 1) * tile_comp->cblk_w > sb->width ? sb->width : (cblk->no_x + 1) * tile_comp->cblk_w;
		cblk->bry = (cblk->no_y + 1) * tile_comp->cblk_h > sb->height ? sb->height : (cblk->no_y + 1)
				* tile_comp->cblk_h;

		cblk->width = cblk->brx - cblk->tlx;
		cblk->height = cblk->bry - cblk->tly;

		cblk->parent_sb = sb;

		//		println_var(INFO, "no:%d no_x:%d no_y:%d tlx:%d tly:%d brx:%d bry:%d w:%d h:%d", cblk->cblk_no, cblk->no_x, cblk->no_y, cblk->tlx, cblk->tly, cblk->brx, cblk->bry, cblk->width, cblk->height);
	}

	//	println_end(INFO);
}

/**
 * @brief Initializes subbands.
 *
 * @param res_lvl
 */
void init_subbands(type_res_lvl *res_lvl) {
	//	println_start(INFO);
	uint8_t i;
	type_subband *sb;
	//	type_subband *sb_ll;
	type_tile *tile;
	type_tile_comp *tile_comp;
	uint8_t xob, yob;
	uint16_t tmp_x, tmp_y;
	uint16_t sb_ll_width, sb_ll_height;

	res_lvl->subbands = (type_subband *) malloc(res_lvl->num_subbands * sizeof(type_subband));

	tile_comp = res_lvl->parent_tile_comp;
	tile = tile_comp->parent_tile;

	sb_ll_width = ((tile->width + (1 << res_lvl->dec_lvl_no) - 1) >> res_lvl->dec_lvl_no);
	sb_ll_height = ((tile->height + (1 << res_lvl->dec_lvl_no) - 1) >> res_lvl->dec_lvl_no);

	//	println_var(INFO, "res_lvl_no:%d dec_lvl_no:%d tlx:%d tly:%d brx:%d bry:%d w:%d h:%d prc_exp_w:%d prc_exp_h:%d num_hprc:%d num_vprc:%d num_prcs:%d num_sbs:%d", res_lvl->res_lvl_no, res_lvl->dec_lvl_no, res_lvl->tlx, res_lvl->tly, res_lvl->brx, res_lvl->bry, res_lvl->width, res_lvl->height, res_lvl->prc_exp_w, res_lvl->prc_exp_h, res_lvl->num_hprc, res_lvl->num_vprc, res_lvl->num_prcs, res_lvl->num_subbands);

	for (i = 0; i < res_lvl->num_subbands; i++) {
		sb = &(res_lvl->subbands[i]);

		if (res_lvl->res_lvl_no == 0) {
			sb->orient = i;
		} else {
			sb->orient = i + 1;
		}

		xob = sb->orient & 1;
		yob = (sb->orient >> 1) & 1;
		//		printf("or:%d xob:%d yob:%d\n", sb->orient, xob, yob);

		tmp_x = ((1 << (res_lvl->dec_lvl_no - 1)) * xob);
		tmp_y = ((1 << (res_lvl->dec_lvl_no - 1)) * yob);
		//		printf("tmp_x:%d tmp_y:%d\n", tmp_x, tmp_y);
		sb->tlx = (tile->tlx - tmp_x + ((1 << res_lvl->dec_lvl_no) - 1)) / (1 << res_lvl->dec_lvl_no);
		sb->tly = (tile->tly - tmp_y + ((1 << res_lvl->dec_lvl_no) - 1)) / (1 << res_lvl->dec_lvl_no);
		sb->brx = (tile->brx - tmp_x + ((1 << res_lvl->dec_lvl_no) - 1)) / (1 << res_lvl->dec_lvl_no);
		sb->bry = (tile->bry - tmp_y + ((1 << res_lvl->dec_lvl_no) - 1)) / (1 << res_lvl->dec_lvl_no);

		sb->width = sb->brx - sb->tlx;
		sb->height = sb->bry - sb->tly;

		sb->tlx += xob * sb_ll_width;
		sb->tly += yob * sb_ll_height;

		sb->brx += xob * sb_ll_width;
		sb->bry += yob * sb_ll_height;

		sb->num_xcblks = (sb->width + (tile_comp->cblk_w - 1)) / tile_comp->cblk_w;
		sb->num_ycblks = (sb->height + (tile_comp->cblk_h - 1)) / tile_comp->cblk_h;
		sb->num_cblks = sb->num_xcblks * sb->num_ycblks;
		//		printf("tlx:%d tly:%d Num cblks:%d Sb w:%d Sb h:%d\n", sb->tlx, sb->tly, sb->num_cblks, sb->width, sb->height);
		sb->parent_res_lvl = res_lvl;

		sb->cblks_data_d = NULL;

		init_codeblocks(sb);
	}
	//	println_end(INFO);
}

/**
 * @brief To save memory the resolution level with no = 0 is not kept separately. LL subband from the resolution level no = 1 equals resolution level no = 0.
 * @param tile_comp
 */
void init_resolution_lvls(type_tile_comp *tile_comp) {
	//	println_start(INFO);
	uint8_t i;
	type_res_lvl *res_lvl;
	type_tile *parent_tile;
	/* n = 2^(Nl-r) */
	uint16_t n;
	/* Precinct width and height */
	int prec_width, prec_height;
	tile_comp->res_lvls = (type_res_lvl *) malloc(tile_comp->num_rlvls * sizeof(type_res_lvl));

	parent_tile = tile_comp->parent_tile;

	//	println_var(INFO, "w:%d h:%d num_dlvls:%d num_rlvls:%d cblk_exp_w:%d cblk_exp_h:%d cblk_w:%d cblk_h:%d", tile_comp->width, tile_comp->height, tile_comp->num_dlvls, tile_comp->num_rlvls, tile_comp->cblk_exp_w, tile_comp->cblk_exp_h, tile_comp->cblk_w, tile_comp->cblk_h);

	for (i = 0; i < tile_comp->num_rlvls; i++) {
		res_lvl = &(tile_comp->res_lvls[i]);

		res_lvl->res_lvl_no = i;
		/* Nl - (r - 1)
		 * if (r - 1 < 0) than return 0 */
		res_lvl->dec_lvl_no = tile_comp->num_dlvls - ((res_lvl->res_lvl_no == 0) ? 0 : (res_lvl->res_lvl_no - 1));

		n = 1 << (tile_comp->num_dlvls - res_lvl->res_lvl_no);
		res_lvl->tlx = (parent_tile->tlx + (n - 1)) / n;
		res_lvl->tly = (parent_tile->tly + (n - 1)) / n;

		res_lvl->brx = (parent_tile->brx + (n - 1)) / n;
		res_lvl->bry = (parent_tile->bry + (n - 1)) / n;

		res_lvl->width = res_lvl->brx - res_lvl->tlx;
		res_lvl->height = res_lvl->bry - res_lvl->tly;
		/* Set to maximum precinct size. One precinct per resolution level */
		res_lvl->prc_exp_w = 15;
		res_lvl->prc_exp_h = 15;

		prec_width = 1 << res_lvl->prc_exp_w;
		prec_height = 1 << res_lvl->prc_exp_h;
		res_lvl->num_vprc = (res_lvl->width + (prec_width - 1)) / prec_width;
		res_lvl->num_hprc = (res_lvl->height + (prec_height - 1)) / prec_height;
		res_lvl->num_prcs = res_lvl->num_vprc * res_lvl->num_hprc;

		if (res_lvl->res_lvl_no == 0) {
			/* LL subband */
			res_lvl->num_subbands = 1;
		} else {
			/* HL, LH, HH subbands */
			res_lvl->num_subbands = 3;
		}

		res_lvl->parent_tile_comp = tile_comp;

		init_subbands(res_lvl);
	}
	//	println_end(INFO);
}

/**
 * @brief Initializes tile components. Allocates memory for them both on the host and the device.
 *
 * @param tile This tile components will be initialized
 */
void init_tile_comps(type_tile *tile, type_parameters *param) {
	uint16_t i;
	type_tile_comp *tile_comp;
	type_image *parent_img;

	println_start(INFO);

	parent_img = tile->parent_img;
	tile->tile_comp = (type_tile_comp *) malloc(parent_img->num_components * sizeof(type_tile_comp));

	//	println_var(INFO, "no:%d tlx:%d tly:%d brx:%d bry:%d w:%d h:%d", tile->tile_no, tile->tlx, tile->tly, tile->brx, tile->bry, tile->width, tile->height);

	for (i = 0; i < parent_img->num_components; i++) {
		uint64_t size_to_allocate;

		//		println_var(INFO, "%d\n", i);
		tile_comp = &(tile->tile_comp[i]);

		tile_comp->tile_comp_no = i;

		/* TODO: to be conformant to the standard there has to be a way to differentiate between tile and tile-component size */
		/* For simplicity we accept that tile size equals tile-component size, but it does not conform ISO standard. */
		tile_comp->width = tile->width;
		tile_comp->height = tile->height;

		tile_comp->num_dlvls = parent_img->num_dlvls;
		tile_comp->num_rlvls = tile_comp->num_dlvls + 1;

		/* Maximum code block size is 64x64 (2^6x2^6).*/
		tile_comp->cblk_exp_w = param->param_cblk_exp_w;
		tile_comp->cblk_exp_h = param->param_cblk_exp_h;
		tile_comp->cblk_w = 1 << tile_comp->cblk_exp_w;
		tile_comp->cblk_h = 1 << tile_comp->cblk_exp_h;
		//printf("tile w:%d h:%d\n", tile_comp->width, tile_comp->height);
		tile_comp->parent_tile = tile;

		size_to_allocate = tile_comp->width * tile_comp->height * sizeof(type_data);
		println_var(INFO, "Allocating %u bytes on both host and device (%u x %u x %u)", size_to_allocate, tile_comp->width, tile_comp->height, sizeof(type_data));

		cuda_h_allocate_mem((void **) &(tile_comp->img_data), size_to_allocate);
		//		tile_comp->img_data = (type_data *) malloc(tile_comp->width * tile_comp->height * sizeof(type_data));
		cuda_d_allocate_mem((void **) &(tile_comp->img_data_d), size_to_allocate);

		init_resolution_lvls(tile_comp);
	}

	println_end(INFO);
}

/**
 * @brief Initializes tiles in the image container. Calculates tiles positions, dimensions and references.
 * Passes this data to init_tile_comps
 *
 * @param _img Container in which tiles should be initialized in.
 */
void init_tiles(type_image **_img, type_parameters *param) {
	//	println_start(INFO);
	type_image *img = *_img;
	uint32_t i = 0;
	/* Horizontal position of the tile */
	uint16_t p;
	/* Vertical position of the tile */
	uint16_t q;
	/* Temporary pointers */
	type_tile *tile;

	println_start(INFO);

	/* Checks if tile width and height are <= image width and height */
	img->tile_h = (param->param_tile_h == -1U ? img->height : (param->param_tile_h <= img->height ?param->param_tile_h : img->height) ); ///Nominal tile height.
	img->tile_w = (param->param_tile_w == -1U ? img->width : (param->param_tile_w <= img->width ? param->param_tile_w : img->width)); ///Nominal tile width.
	//println_var(INFO, "container->tile_h:%d container->tile_w:%d", container->tile_h, container->tile_w);

	img->num_xtiles = (img->width + (img->tile_w - 1)) / img->tile_w;
	img->num_ytiles = (img->height + (img->tile_h - 1)) / img->tile_h;
	img->num_tiles = img->num_xtiles * img->num_ytiles;

	//	cuda_h_allocate_mem((void **) &(img->tile), img->num_tiles * sizeof(type_tile));
	img->tile = (type_tile *) malloc(img->num_tiles * sizeof(type_tile));

	println_var(INFO, "w:%d h:%d no_com:%d area:%d t_w:%d t_h:%d t_x:%d t_y:%d no_t:%d", img->width,img->height,img->num_components,img->area_alloc,img->tile_w,img->tile_h,img->num_xtiles,img->num_ytiles,img->num_tiles);

	for (i = 0; i < img->num_tiles; i++) {
		tile = &(img->tile[i]);
		tile->tile_no = i;

		p = tile->tile_no % img->num_xtiles;
		q = tile->tile_no / img->num_xtiles;

		tile->tlx = p * img->tile_w;
		tile->tly = q * img->tile_h;

		tile->brx = (p + 1) * img->tile_w > img->width ? img->width : (p + 1) * img->tile_w;
		tile->bry = (q + 1) * img->tile_h > img->height ? img->height : (q + 1) * img->tile_h;

		tile->width = tile->brx - tile->tlx;
		tile->height = tile->bry - tile->tly;

		tile->parent_img = img;

		init_tile_comps(tile, param);
	}

	println_end(INFO);
}

/**
 * @brief Set coding parameters.
 *
 * @param img
 */
void set_coding_parameters(type_image *img, type_parameters *param) {
	img->coding_param = (type_coding_param *) malloc(sizeof(type_coding_param));
	img->coding_param->imgarea_tlx = 0;
	img->coding_param->imgarea_tly = 0;
	img->coding_param->imgarea_width = img->width;
	img->coding_param->imgarea_height = img->height;
	img->coding_param->tilegrid_tlx = 0;
	img->coding_param->tilegrid_tly = 0;
	img->coding_param->comp_step_x = 1;
	img->coding_param->comp_step_y = 1;
	img->coding_param->base_step = 1.0 / (float) (1 << (img->num_range_bits - 1));
	if (param->param_target_size > 0) {
		img->coding_param->target_size = param->param_target_size;
	} else {
		/** Size in bytes */
		img->coding_param->target_size = (param->param_bp * img->width * img->height * img->num_components) / 8.0;
	}
}

static void init_params(type_image *img, type_parameters *param) {
	img->wavelet_type = param->param_wavelet_type;
	img->use_mct = param->param_use_mct;
	img->use_part2_mct = param->param_use_part2_mct;
	img->mct_compression_method = param->param_mct_compression_method;

	img->coding_style = CODING_STYLE;
	img->prog_order = LY_RES_COMP_POS_PROG;
	img->num_layers = NUM_LAYERS;
}

/**
 * @brief Read image from file.
 *
 * @param path Path to image file.
 * @param _container Pointer to a malloc'ed memory for type_image.
 */
int read_image(type_image *img, type_parameters *param)
{
	init_params(img, param);

	/* Read input image */
	if(img->bsq_file || img->bip_file || img->bil_file)
	{
		if(read_hyper_image(&img, param) == -1)
			return -1;
//		println_var(INFO, "Bsq image type!");
	} else
	{
		read_ordinary_image(&img, param);
//		println_var(INFO, "Ordinary image type!");
	}

	return 0;
}

void save_image(type_image *img) {
	if(img->bsq_file == 1)
	{
		save_img_hyper(img, img->out_file);
	} else
	{
		save_img_ord(img, img->out_file);
	}
}
