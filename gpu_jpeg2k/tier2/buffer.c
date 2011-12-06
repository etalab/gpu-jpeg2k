/*
 * buffer.c
 *
 *  Created on: Dec 1, 2011
 *      Author: miloszc
 */

#include "buffer.h"

void init_dec_buffer(FILE *fsrc, type_buffer *src_buff) {
	fseek(fsrc, 0, SEEK_END);
	long file_length = ftell(fsrc);
	fseek(fsrc, 0, SEEK_SET);

	src_buff->data = (uint8_t *) malloc(file_length);
	src_buff->size = file_length;

	fread(src_buff->data, 1, file_length, fsrc);

	src_buff->start = src_buff->data;
	src_buff->end = src_buff->data + src_buff->size;
	src_buff->bp = src_buff->data;
	src_buff->bits_count = 0;
	src_buff->byte = 0;
}
