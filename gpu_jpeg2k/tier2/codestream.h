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
 * @file codestream.h
 *
 * @author Milosz Ciznicki
 */

#ifndef CODESTREAM_H_
#define CODESTREAM_H_

#include <assert.h>

#include "../types/image_types.h"
#include "../types/buffered_stream.h"

typedef struct type_packet type_packet;

/** Packet parameters */
struct type_packet{
	uint16_t *inclusion;
	uint16_t *zero_bit_plane;
	uint16_t *num_coding_passes;
};

void encode_codestream(type_buffer *buffer, type_image *img);
void decode_codestream(type_buffer *buffer, type_image *img);

#endif /* CODESTREAM_H_ */
