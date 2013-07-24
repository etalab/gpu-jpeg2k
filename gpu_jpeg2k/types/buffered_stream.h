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
 * @file buffered_stream.h
 *
 * @author Milosz Ciznicki
 */

#ifndef BUFFERED_STREAM_H_
#define BUFFERED_STREAM_H_

#include <stdio.h>
#include <stdint.h>

#define INIT_BUF_SIZE 128

typedef struct type_buffer type_buffer;

struct type_buffer {
	unsigned char *data;
	uint32_t byte;
	unsigned int bytes_count;
	int bits_count;
	unsigned long int size;
	unsigned char *bp;
	unsigned char *start;
	unsigned char *end;
};

void init_buffer(type_buffer *buffer);
void enlarge_buffer(type_buffer *buffer);
void write_byte(type_buffer *buffer, uint8_t val);
void write_short(type_buffer *buffer, uint16_t val);
void write_int(type_buffer *buffer, uint32_t val);
void bit_stuffing(type_buffer *buffer);
void write_stuffed_byte(type_buffer *buffer);
void write_zero_bit(type_buffer *buffer);
void write_one_bit(type_buffer *buffer);
void write_bits(type_buffer *buffer, int bits, int n);
void write_array(type_buffer *buffer, uint8_t *in, int length);
void update_buffer_byte(type_buffer *buffer, int pos, uint8_t val);
void write_buffer_to_file(type_buffer *buffer, FILE *fp);
uint32_t read_buffer(type_buffer *buffer, int n);
uint32_t read_bits(type_buffer *buffer, int n);
uint32_t inalign(type_buffer *buffer);
uint16_t peek_marker(type_buffer *buffer);
uint8_t read_byte(type_buffer *buffer);

#endif /* BUFFERED_STREAM_H_ */
