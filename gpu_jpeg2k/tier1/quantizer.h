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
