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
