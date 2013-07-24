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
 * @file image_mct.h
 *
 * @author Kamil Balwierz
 */

#ifndef IMAGE_MCT_H_
#define IMAGE_MCT_H_

#define MCT_8BIT_INT 0
#define MCT_16BIT_INT 1
#define MCT_32BIT_FLOAT 2
#define MCT_64BIT_DOUBLE 3
#define MCT_128BIT_DOUBLE 4 /* actually used only within wavelet context in ATK segment marker */

#define MCT_DECORRELATION_TRANSFORMATION 0
#define MCT_DEPENDENCY_TRANSFORMATION 1
#define MCT_DECORRELATION_OFFSET 2
#define MCT_DEPENDENCY_OFFSET 3

#define MCC_MATRIX_BASED 0
#define MCC_WAVELET_BASED_LOW 2
#define MCC_WAVELET_BASED_HIGH 3

typedef struct type_mct type_mct;
typedef struct type_mcc type_mcc;
typedef struct type_mcc_data type_mcc_data;
typedef struct type_mic type_mic;
typedef struct type_mic_data type_mic_data;
typedef struct type_atk type_atk;
typedef struct type_ads type_ads;
typedef struct type_multiple_component_transformations type_multiple_component_transformations;

/** Data gathering point for multiple component transformation as in 15444-2 Annex I */ 
struct type_multiple_component_transformations
{
	/** Transformation matrices */
	type_mct* mcts[4];
	/** Number of transformation matrices by type */
	uint8_t mcts_count[4];

	/** Multiple component collection segments */
	type_mcc* mccs;

	/** Count of component collection segments */
	uint8_t mccs_count;

	/** Multiple intermediate collection segments */
	type_mic* mics;

	/** Count of intermediate collection segments */
	uint8_t mics_count;

	/** Arbitrary decomposition styles */
	type_ads* adses;

	/** Count of ADS segments */
	uint8_t ads_count;

	/** Arbitrary transformation kernels */
	type_atk* atks;

	/** Count of ATK segemnts */
	uint8_t atk_count;
};

/** MCT as in 15444-2 Annex A.3.7 */
struct type_mct
{
	/** Matrix definition index */
	uint8_t index;

	/** Matrix type */
	uint8_t type;

	/** Matrix element data type */
	uint8_t element_type;

	/** Element count */
	uint32_t length;

	/** Data */
	uint8_t* data;
};

/** MCC as in 15444-2 Annex A.3.8 */
struct type_mcc {
	/** Index of marker segment */
	uint8_t index;

	/** Count of collections in segment */
	uint8_t count;

	/** Component collections */
	type_mcc_data* data;
};

/** Component Collection part of MCC segment as in 15444-2 Annex A.3.8 */
struct type_mcc_data {
	/** Decorrelation type */
	uint8_t type;

	/** Number of input components */
	uint16_t input_count;

	/** Are components number 8 or 16 bit? */
	uint8_t input_component_type;

	/** Input components identifiers */
	uint8_t* input_components;

	/** Number of output components */
	uint16_t output_count;

	/** Are components number 8 or 16 bit? */
	uint8_t output_component_type;

	/** Input components identifiers */
	uint8_t* output_components;

	/** Number of transform matrix to use in decorrelation process 
	 *		Used only with matrix based decorrelation!
	 */
	uint8_t decorrelation_transform_matrix;

	/** Number of transform offset matrix to use in decorrelation process 
	 *		Used only with matrix based decorrelation!
	 */
	uint8_t deccorelation_transform_offset;

	/** Index of ATK marker
	 * 	Used only with wavelet based decorrelation!
	 */
	uint8_t atk;

	/** Index of ADS marker
	 * 	Used only with wavelet based decorrelation!
	 */
	uint8_t ads;
};

/** MIC as in 15444-2 Annex A.3.9 */
struct type_mic {
	/** Index of marker segment */
	uint8_t index;

	/** Count of collections in segment */
	uint8_t count;

	/** Component collections */
	type_mic_data* data;
};

/** Component Intermediate Collection part of MIC segment as in 15444-2 Annex A.3.9 */
struct type_mic_data {
	/** Number of input components */
	uint16_t input_count;

	/** Are components number 8 or 16 bit? */
	uint8_t input_component_type;

	/** Input components identifiers */
	uint8_t* input_components;

	/** Number of output components */
	uint16_t output_count;

	/** Are components number 8 or 16 bit? */
	uint8_t output_component_type;

	/** Input components identifiers */
	uint8_t* output_components;

	/** Number of transform matrix to use in decorrelation process */ 
	uint8_t decorrelation_transform_matrix;

	/** Number of transform offset matrix to use in decorrelation process */ 
	uint8_t deccorelation_transform_offset;
};

struct type_ads {
	/** Index of marker segment */
	uint8_t index;
	
	/** Number of elements in the string defining the number of decomposition sub-levels */
	uint8_t IOads;

	/** String defining the number of decomposition sub-levels. */
	uint8_t* DOads;

	/** Number of elements in the string defining the arbitrary decomposition structure. */
	uint8_t ISads;

	/** String defining the arbitrary decomposition structure. */
	uint8_t* DSads;
};

struct type_atk {
	/** Index of marker segment */
	uint8_t index;

	/** Coefficients data type */
	uint8_t coeff_type;

	/** Wavelet filters */
	uint8_t filter_category;

	/** Wavelet type */
	uint8_t wavelet_type;

	/** Odd/Even indexed subsequence */
	uint8_t m0;

	/** Number of lifting steps */
	uint8_t lifing_steps;

	/** Number of lifting coefficients at lifting step */
	uint8_t lifting_coefficients_per_step;

	/** Offset for lifting step */
	uint8_t lifting_offset;

	/** Base two scaling exponent for lifting step s, εs for the reversible transform only */
	uint8_t* scaling_exponent;
	
	/** Scaling factor, for the irreversible transform only*/
	uint8_t* scaling_factor;

	/**The ith lifting coefficient for the jth lifting step,αs,k. The index, i, ranges from i = 0 to Natk-1 and is the inner loop (present for all of j). The index, j, ranges from j = 0 to Latk-1 and is the outer loop(incremented after a full run of i). */
	uint8_t * coefficients;

	/** The ith additive residue for lifting step, s. The index, i, ranges from i = 0 to Natk-1. Present for reversible transformations */
	uint8_t* additive_residue;
};

#endif
