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
 * @file dwt.h
 *
 * @author Milosz Ciznicki
 */


#ifndef DWT_H_
#define DWT_H_

#include "../types/image_types.h"

#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif

#define SHIFT 4
#define ADD_VALUES 7

#define MEMSIZE 40
#define PATCHX 32
#define PATCHY 32
#define PATCHX_DIV_2 16
#define PATCHY_DIV_2 16
#define BLOCKSIZEX 16
#define BLOCKSIZEY 16

#define OFFSET_97 4
#define OFFSET_53 2
#define IOFFSET_97 3
#define IOFFSET_53 2
#define OFFSETX_DIV_2 2
#define OFFSETY_DIV_2 2
#define FIRST_BLOCK 0

extern void fwt(type_tile *tile);
extern  void iwt(type_tile *tile);
extern void fwt_dbg(type_tile *tile);
void save_tile_comp_no_dbg(type_tile_comp *tile_comp, int i);
void save_tile_comp_no_shift_dbg(type_tile_comp *tile_comp, int i);
void print_tile_comp(type_tile_comp *tile_comp);
void save_tile_comp_with_shift(type_tile_comp *tile_comp, char *filename, int shift);

#endif /* DWT_H_ */
