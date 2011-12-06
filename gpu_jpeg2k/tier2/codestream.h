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
