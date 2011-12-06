/**
 * @file quantization.h
 *
 * @author milosz 
 * @date 06-09-2010
 */

#ifndef QUANTIZER_H_
#define QUANTIZER_H_

#include <stdio.h>

#include "quantization.h"

extern void quantize_tile(type_tile *tile);
extern void quantize_tile_dbg(type_tile *tile);
extern type_subband *quantization(type_subband *sb);

#endif /* QUANTIZER_H_ */
