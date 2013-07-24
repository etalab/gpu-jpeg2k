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
 * @file parameters.h
 *
 * @brief All input parameters of the application.
 *
 * This file serves as a db of the default values of the parameters and as a connector between effector functions and setter functions
 *  (those which set parameters according to values supplied by the user).
 *
 * @author Jakub Misiorny <misiorny@man.poznan.pl>
 */


#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <stdint.h>

typedef struct type_parameters type_parameters;

/** Image parameters */
struct type_parameters {
	uint16_t param_tile_w; /// Tile width. According to this and param_tile_height all the parameters of the tiles are set. -1 is no tiling (only one tile which covers entire image).
	uint16_t param_tile_h; /// Tile height. According to this and param_tile_width all the parameters of the tiles are set. -1 is no tiling (only one tile which covers entire image).
	uint8_t param_tile_comp_dlvls;
	uint8_t param_cblk_exp_w; ///Maximum codeblock size is 2^6 x 2^6 ( 64 x 64 ).
	uint8_t param_cblk_exp_h; ///Maximum codeblock size is 2^6 x 2^6 ( 64 x 64 ).
	uint8_t param_wavelet_type; ///Lossy encoding
	uint8_t param_use_mct;//Multi-component transform
	uint8_t param_device;//Which device use
	uint32_t param_target_size;//Target size of output file
	float param_bp;//Bits per pixel per component
	uint8_t param_use_part2_mct; // Multiple component transform as in 15444-2
	uint8_t param_mct_compression_method; // 0 klt 2 wavelet
	uint32_t param_mct_klt_iterations; // max number of iterations of Gram-Schmidt algorithm
	float param_mct_klt_border_eigenvalue; // cut-off for dumping components 
	float param_mct_klt_err; // error sufficient for Gram-Schmit algorithm to end iteration
};


int parse_config(const char *filename, type_parameters *param);
void default_config_values(type_parameters *param);

#endif /* PARAMETERS_H_ */
