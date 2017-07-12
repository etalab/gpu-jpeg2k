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
 * @file codestream_mct.c
 *
 * @author Kamil Balwierz
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "codestream.h"
#include "markers.h"
#include "../types/image_types.h"
#include "../print_info/print_info.h"
#include "codestream_mct.h"


/**
 * @brief Writes MCT marker.
 *
 * @param buffer
 * @param img
 */
void write_mct_marker(type_buffer *buffer, type_mct *mct) {
	int length;
	uint8_t Smct;
	int i;
	
	write_short(buffer, MCT);
	length = 5 + mct->length*(1<<mct->element_type);
	write_int(buffer, length);
	Smct = (mct->element_type << 6);
	Smct |= (mct->type << 4);
	Smct |= mct->index;
	write_byte(buffer, Smct);
	
	for(i=0; i<length-5; ++i) {
		write_byte(buffer, mct->data[i]);
	}
}

/**
 * @brief Reads MCT marker.
 *
 * @param buffer
 * @param img
*/ 
void read_mct_marker(type_buffer *buffer, type_image *img) {
	int marker;
	int length;
	uint16_t Smct;
	uint8_t type;
	int i;

	/* Read MCT Marker */
	marker = read_buffer(buffer, 2);
	if(marker != MCT)
	{
		println_var(INFO, "Error: Expected MCT marker instead of %x", marker);
	}

	length = read_buffer(buffer, 4)-5;

	/* Read Smct */
	Smct = read_byte(buffer);
	
	type = (Smct&(3<<4))>>4;

	type_mct* old_mcts = img->mct_data->mcts[type];
	img->mct_data->mcts[type] = (type_mct*)realloc(img->mct_data->mcts[type], sizeof(type_mct) * (++img->mct_data->mcts_count[type]));
	type_mct* mct = &img->mct_data->mcts[type][img->mct_data->mcts_count[type]-1];
	
	if(img->mct_data->mcts[type] == NULL) {
		img->mct_data->mcts[type] = old_mcts;
		--img->mct_data->mcts_count[type];
		println_var(INFO, "Error: Memory reallocation failed! Ignoring MCT with Smct %u Skipping data", Smct);
		for(i=0; i<length-1; ++i) {
			read_byte(buffer);
		}
	} else {
		mct->index = Smct&0x0F;
		mct->type = type;
		mct->element_type = (Smct&(3<<6))>>6;
		mct->length = length/(1<<mct->element_type);
		mct->data = (uint8_t*)my_malloc(length);
		for(i=0; i<length; ++i) {
			mct->data[i] = read_byte(buffer);
		}
	}
}

/**
 * @brief Writes MMC marker.
 *
 * @param buffer
 * @param img
 */
void write_mcc_marker(type_buffer *buffer, type_mcc *mcc) {
	int length;
	int i,j;
	
	write_short(buffer, MCC);
	length=5 + 6 * mcc->count;

	for(i=0; i<mcc->count; ++i) {
		length += mcc->data[i].input_count * (1 << mcc->data[i].input_component_type) + mcc->data[i].output_count * ( 1 << mcc->data[i].output_component_type);
	}

	write_int(buffer, length);
	write_byte(buffer, mcc->index);

	type_mcc_data* data;

	for(i=0; i<mcc->count; ++i) {
		data = &mcc->data[i];
		write_byte(buffer, data->type);
		write_short(buffer, data->input_count | ( data->input_component_type << 14));
		for(j=0; j<data->input_count * (1 << data->input_component_type); ++j) {
			write_byte(buffer, data->input_components[j]);
		}
		write_short(buffer, data->output_count | ( data->output_component_type << 14));
		for(j=0; j<data->output_count * (1 << data->output_component_type); ++j) {
			write_byte(buffer, data->output_components[j]);
		}
		if(data->type & 2) {
			write_byte(buffer, data->atk | (data->ads << 4));
		} else {
			write_byte(buffer, data->decorrelation_transform_matrix | ( data->deccorelation_transform_offset << 4));
		}
	}
}

/**
 * @brief Reads MMC marker.
 *
 * @param buffer
 * @param img
*/ 
void read_mcc_marker(type_buffer *buffer, type_image *img) {
	int marker;
	int length;
	uint16_t i;
	int count=0;
	uint16_t temp_16;
	uint8_t temp_8;

	type_mcc* old_mccs = img->mct_data->mccs;
	img->mct_data->mccs = (type_mcc*)realloc(img->mct_data->mccs, sizeof(type_mcc) * (++img->mct_data->mccs_count));
	type_mcc* mcc = &img->mct_data->mccs[img->mct_data->mccs_count-1];
	mcc->data = NULL;

	type_mcc_data* mcc_data = NULL;

	if(img->mct_data->mccs == NULL) {
		img->mct_data->mccs = old_mccs;
		--img->mct_data->mccs_count;
		println_var(INFO, "Error: Memory reallocation failed! Aborting read of MCC segment");
		return;
	}

	/* Read MCC Marker */
	marker = read_buffer(buffer, 2);
	if(marker != MCC)
	{
		println_var(INFO, "Error: Expected MCC marker instead of %x", marker);
	}

	length = read_buffer(buffer, 4)-5;
	mcc->index = read_byte(buffer);
	/* reading unknown number of component collections */ 
	while(length>0) {
		mcc->data = (type_mcc_data*)realloc((void*)mcc->data, sizeof(type_mcc_data)*(++count));
		mcc_data = &mcc->data[count-1];

		/* input component collection header */
		mcc_data->type = read_byte(buffer)&3;
		temp_16 = read_buffer(buffer,2);
		mcc_data->input_count = temp_16 & 0x1FFF;
		mcc_data->input_component_type = temp_16 >> 14;
		length-=3;

		/* input component collection data */
		temp_16 = mcc_data->input_count * (1<<mcc_data->input_component_type);
		length-=temp_16;
		mcc_data->input_components = (uint8_t*)calloc(temp_16, sizeof(uint8_t));
		for(i=0; i<temp_16; ++i) {
			mcc_data->input_components[i] = read_byte(buffer);
		}

		/* output component collection header */
		temp_16 = read_buffer(buffer,2);
		mcc_data->output_count = temp_16 & 0x1FFF;
		mcc_data->output_component_type = temp_16 >> 14;
		length-=2;

		/* output component collection data */
		temp_16 = mcc_data->output_count * (1<<mcc_data->output_component_type);
		length-=temp_16;
		mcc_data->output_components = (uint8_t*)calloc(temp_16, sizeof(uint8_t));
		for(i=0; i<temp_16; ++i) {
			mcc_data->output_components[i] = read_byte(buffer);
		}

		/* component collection footer */
		temp_8 = read_byte(buffer);
		if(mcc_data->type & 2) {
			/* wavelet based decorrelation */
			mcc_data->atk = temp_8 & 0xF;
			mcc_data->ads = temp_8 >> 4;
			mcc_data->decorrelation_transform_matrix = 0x0;
			mcc_data->deccorelation_transform_offset = 0x0;
		} else {
			/* matrix based decorrelation */
			mcc_data->atk = 0x0;
			mcc_data->ads = 0x0;
			mcc_data->decorrelation_transform_matrix = temp_8 & 0xF;
			mcc_data->deccorelation_transform_offset = temp_8 >> 4;
		}
		length-=1;

	}
	mcc->count = count;
}

/**
 * @brief Writes MIC marker.
 *
 * @param buffer
 * @param img
 */
void write_mic_marker(type_buffer *buffer, type_mic *mic) {
	int length;
	int i,j;
	
	write_short(buffer, MIC);
	length=5 + 5 * mic->count;

	for(i=0; i<mic->count; ++i) {
		length += mic->data[i].input_count * (1 << mic->data[i].input_component_type) + mic->data[i].output_count * ( 1 << mic->data[i].output_component_type);
	}

	write_int(buffer, length);
	write_byte(buffer, mic->index);

	type_mic_data* data;

	for(i=0; i<mic->count; ++i) {
		data = &mic->data[i];
		write_short(buffer, data->input_count | ( data->input_component_type << 14));
		for(j=0; j<data->input_count * (1 << data->input_component_type); ++j) {
			write_byte(buffer, data->input_components[0]);
		}
		write_short(buffer, data->output_count | ( data->output_component_type << 14));
		for(j=0; j<data->output_count * (1 << data->output_component_type); ++j) {
			write_byte(buffer, data->output_components[0]);
		}
		write_byte(buffer, data->decorrelation_transform_matrix | ( data->deccorelation_transform_offset << 4));
	}
}

/**
 * @brief Reads MIC marker.
 *
 * @param buffer
 * @param img
*/ 
void read_mic_marker(type_buffer *buffer, type_image *img) {
	int marker;
	int length;
	uint16_t i;
	int count=0;
	uint16_t temp_16;
	uint8_t temp_8;

	type_mic* old_mics = img->mct_data->mics;
	img->mct_data->mics = (type_mic*)realloc(img->mct_data->mics, sizeof(type_mic) * (++img->mct_data->mics_count));
	type_mic* mic = &img->mct_data->mics[img->mct_data->mics_count-1];

	if(img->mct_data->mics == NULL) {
		img->mct_data->mics = old_mics;
		--img->mct_data->mics_count;
		println_var(INFO, "Error: Memory reallocation failed! Aborting read of MIC segment");
		return;
	}

	type_mic_data* mic_data;

	/* Read MCC Marker */
	marker = read_buffer(buffer, 2);
	if(marker != MIC)
	{
		println_var(INFO, "Error: Expected MIC marker instead of %x", marker);
	}

	length = read_buffer(buffer, 4)-5;
	mic->index = read_byte(buffer);
	/* reading unknown number of component collections */ 
	while(length>0) {
		mic->data = (type_mic_data*)realloc((void*)mic->data, sizeof(type_mic_data)*++count);
		mic_data = &mic->data[count-1];

		/* input component collection header */
		temp_16 = read_buffer(buffer,2);
		mic_data->input_count = temp_16 & 0x1FFF;
		mic_data->input_component_type = temp_16 >> 14;
		length-=2;

		/* input component collection data */
		temp_16 = mic_data->input_count * (1<<mic_data->input_component_type);
		length-=temp_16;
		mic_data->input_components = (uint8_t*)calloc(temp_16, sizeof(uint8_t));
		for(i=0; i<temp_16; ++i) {
			mic_data->input_components[i] = read_byte(buffer);
		}

		/* output component collection header */
		temp_16 = read_buffer(buffer,2);
		mic_data->output_count = temp_16 & 0x1FFF;
		mic_data->output_component_type = temp_16 >> 14;
		length-=2;

		/* output component collection data */
		temp_16 = mic_data->output_count * (1<<mic_data->output_component_type);
		length-=temp_16;
		mic_data->output_components = (uint8_t*)calloc(temp_16, sizeof(uint8_t));
		for(i=0; i<temp_16; ++i) {
			mic_data->output_components[i] = read_byte(buffer);
		}

		/* component collection footer */
		temp_8 = read_byte(buffer);
		mic_data->decorrelation_transform_matrix = temp_8 & 0xF;
		mic_data->deccorelation_transform_offset = temp_8 >> 4;
		length-=1;
	}
	mic->count = count;
}

/**
 * @brief Writes ATK marker.
 *
 * @param buffer
 * @param img
 */
void write_atk_marker(type_buffer *buffer, type_atk *atk) {
	int length;
	int i;
	int count;
	uint16_t Satk;
	
	write_short(buffer, ATK);
	if(atk->wavelet_type == 1) {
		length = 8 + atk->coeff_type * ( atk->lifing_steps * atk->lifting_coefficients_per_step +  atk->lifing_steps);
	} else {
		length = 7 + atk->coeff_type * ( atk->lifing_steps * atk->lifting_coefficients_per_step + 1);
	}
	write_int(buffer, length);
	
	Satk = atk->index;
	Satk &= (atk->coeff_type << 4);
	Satk &= (atk->filter_category << 7);
	Satk &= (atk->wavelet_type << 9);
	Satk &= (atk->m0 << 10);

	write_short(buffer, Satk);
	write_byte(buffer, atk->lifing_steps);
	write_byte(buffer, atk->lifting_coefficients_per_step);
	write_byte(buffer, atk->lifting_offset);

	uint8_t* data = atk->wavelet_type==0?atk->scaling_factor:atk->scaling_exponent;
	for(i=0; i<(1<<atk->coeff_type); ++i) {
		write_byte(buffer, data[i]);
	}

	count = atk->lifing_steps * atk->lifting_coefficients_per_step * atk->wavelet_type?1:(1<<atk->coeff_type);
	for(i=0; i<count; ++i) {
		write_byte(buffer, atk->coefficients[i]);
	}

	if(atk->wavelet_type == 1) {
		count = atk->lifing_steps * (1<<atk->coeff_type);
		for(i=0; i<count; ++i) {
			write_byte(buffer, atk->additive_residue[i]);
		}
	}
}

/**
 * @brief Reads ATK marker.
 *
 * @param buffer
 * @param img
*/ 
void read_atk_marker(type_buffer *buffer, type_image *img) {
	int marker;
	int length;
	int i;
	int count;
	uint16_t temp;
	type_atk* old_atks = img->mct_data->atks;
	img->mct_data->atks = (type_atk*)realloc(img->mct_data->atks, sizeof(type_atk) * (++img->mct_data->atk_count));
	type_atk* atk = &img->mct_data->atks[img->mct_data->atk_count-1];

	if(img->mct_data->atks == NULL) {
		img->mct_data->atks = old_atks;
		--img->mct_data->atk_count;
		println_var(INFO, "Error: Memory reallocation failed! Aborting read of ATK segment");
		return;
	}


	/* Read ATK Marker */
	marker = read_buffer(buffer, 2);
	if(marker != ATK)
	{
		println_var(INFO, "Error: Expected ATK marker instead of %x", marker);
	}
	length = read_buffer(buffer, 4)-5;

	temp = read_byte(buffer);
	atk->index = temp & 0xF;
	atk->coeff_type = (temp & 0x0070) >> 4;
	atk->filter_category = (temp & 0x0180) >> 7;
	atk->wavelet_type = (temp & 0x0200) >> 9;
	atk->m0 = (temp & 0x400) >> 10;

	atk->lifing_steps = read_byte(buffer);
	atk->lifting_coefficients_per_step = read_byte(buffer);
	atk->lifting_offset = read_byte(buffer);

	count = 1<<atk->coeff_type;
	uint8_t* t = (uint8_t*)calloc(count, sizeof(uint8_t));
	for(i=0; i<count; ++i) {
		t[i]=read_byte(buffer);
	}

	if(atk->wavelet_type == 1) {
		atk->scaling_exponent = t;
	} else {
		atk->scaling_factor = t;
	}

	count = atk->lifing_steps * atk->lifting_coefficients_per_step * atk->wavelet_type?1:(1<<atk->coeff_type);
	atk->coefficients = (uint8_t*)calloc(count, sizeof(uint8_t));
	for(i=0; i<count; ++i) {
		atk->coefficients[i] = read_byte(buffer);
	}

	if(atk->wavelet_type == 1) {
		count = atk->lifing_steps * (1<<atk->coeff_type);
		atk->additive_residue = (uint8_t*)calloc(count, sizeof(uint8_t));
		for(i=0; i<count; ++i) {
			atk->additive_residue[i] = read_byte(buffer);
		}
	}
}


/**
 * @brief Writes ADS marker.
 *
 * @param buffer
 * @param img
 */
void write_ads_marker(type_buffer *buffer, type_ads *ads) {
	int length;
	int i;
	uint8_t temp;

	write_short(buffer, ADS);
	length = 5 + (int)ceil((double)(ads->IOads + ads->ISads)/4.0);
	write_short(buffer, length);
	write_byte(buffer, ads->IOads);
	
	i = 0;
	temp = 0;
	while(ads->DOads - i >0) {
		temp |= ads->DOads[i] << ((3-i)*2);
		if(i%4 == 3) {
			write_byte(buffer, temp);
			temp = 0;
		}
		++i;
	}
	if(i%4 != 0) 
		write_byte(buffer, temp);

	write_byte(buffer, ads->ISads);
	i = 0;
	temp = 0;
	while(ads->DSads - i >0) {
		temp |= ads->DSads[i] << ((3-i)*2);
		if(i%4 == 3) {
			write_byte(buffer, temp);
			temp = 0;
		}
		++i;
	}
	if(i%4 != 0) 
		write_byte(buffer, temp);


}

/**
 * @brief Reads ADS marker.
 *
 * @param buffer
 * @param img
*/ 
void read_ads_marker(type_buffer *buffer, type_image *img) {
	int marker;
	int length;
	int i;
	uint8_t temp;

	type_ads* old_adses = img->mct_data->adses;
	img->mct_data->adses = (type_ads*)realloc(img->mct_data->adses, sizeof(type_ads) * (++img->mct_data->ads_count));
	type_ads* ads = &img->mct_data->adses[img->mct_data->ads_count-1];

	if(img->mct_data->adses == NULL) {
		img->mct_data->adses = old_adses;
		--img->mct_data->ads_count;
		println_var(INFO, "Error: Memory reallocation failed! Aborting read of ADS segment");
		return;
	}


	/* Read ADS Marker */
	marker = read_buffer(buffer, 2);
	if(marker != ADS)
	{
		println_var(INFO, "Error: Expected ADS marker instead of %x", marker);
	}
	length = read_buffer(buffer, 4)-5;
	ads->index = read_byte(buffer);
	ads->IOads = read_byte(buffer);
	ads->DOads = (uint8_t*)calloc(ads->IOads, sizeof(uint8_t));	
	i = 0;
	while(ads->DOads - i >0) {
		if(i%4 == 0)
			temp = read_byte(buffer);
		ads->DOads[i] = (temp & (3 << (3-i)*2)) >> (3-i)*2;
		++i;
	}

	ads->ISads = read_byte(buffer);
	ads->DSads = (uint8_t*)calloc(ads->ISads, sizeof(uint8_t));	
	i = 0;
	while(ads->DSads - i >0) {
		if(i%4 == 0)
			temp = read_byte(buffer);
		ads->DSads[i] = (temp & (3 << (3-i)*2)) >> (3-i)*2;
		++i;
	}
}

/** Writes all data needed for performing multiple component transformations as in 15444-2 into codestream */
void write_multiple_component_transformations(type_buffer *buffer, type_image *img) {
	int i,j;
	for(i=0; i < img->mct_data->ads_count; ++i) {
		write_ads_marker(buffer, &img->mct_data->adses[i]);
	}

	for(i=0; i < img->mct_data->atk_count; ++i) {
		write_atk_marker(buffer, &img->mct_data->atks[i]);
	}

	for(i=0; i < 4; ++i) {
		for(j=0; j<img->mct_data->mcts_count[i]; ++j) {
			write_mct_marker(buffer, &img->mct_data->mcts[i][j]);
		}
	}

	for(i=0; i< img->mct_data->mccs_count; ++i) {
		write_mcc_marker(buffer, &img->mct_data->mccs[i]);
	}

	for(i=0; i< img->mct_data->mics_count; ++i) {
		write_mic_marker(buffer, &img->mct_data->mics[i]);
	}
}

/** Reads all data needed for performing multiple component transformations as in 15444-2 from codestream */
void read_multiple_component_transformations(type_buffer *buffer, type_image *img) {
	img->mct_data = (type_multiple_component_transformations*)calloc(1, sizeof(type_multiple_component_transformations));
	while(peek_marker(buffer)==ADS) {
		read_ads_marker(buffer, img);
	}
	while(peek_marker(buffer)==ATK) {
		read_atk_marker(buffer, img);
	}
	while(peek_marker(buffer)==MCT) {
		read_mct_marker(buffer, img);
	}
	while(peek_marker(buffer)==MCC) {
		read_mcc_marker(buffer, img);
	}
	while(peek_marker(buffer)==MIC) {
		read_mic_marker(buffer, img);
	}
}


