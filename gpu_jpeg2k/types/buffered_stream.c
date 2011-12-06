/**
 * @file buffered_stream.c
 *
 * @author Milosz Ciznicki
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "buffered_stream.h"
#include "../print_info/print_info.h"

/**
 * Initializes buffer with INIT_BUF_SIZE.
 *
 * @param buffer
 */
void init_buffer(type_buffer *buffer)
{
//	println_start(INFO);
	buffer->data = (uint8_t*) malloc(INIT_BUF_SIZE);
	buffer->byte = 0;
	buffer->bytes_count = 0;
	buffer->bits_count = 0;
	buffer->size = INIT_BUF_SIZE;
//	println_end(INFO);
}

void seek_buffer(type_buffer *buffer, int pos)
{
	buffer->bp = buffer->start + pos;
}

int32_t num_bytes_left(type_buffer *buffer)
{
	return buffer->end - buffer->bp;
}

uint8_t *get_bp_buffer(type_buffer *buffer)
{
	return buffer->bp;
}

uint8_t read_byte(type_buffer *buffer)
{
	if(buffer->bp >= buffer->end)
	{
		println_var(INFO, "Error: Exceeded buffer bounds!");
		exit(0);
	}
	return *buffer->bp++;
}

uint32_t read_buffer(type_buffer *buffer, int n)
{
	int i;
	uint32_t val = 0;

	for(i = n - 1; i >= 0; i--)
	{
		val += read_byte(buffer) << (i << 3);
	}
	return val;
}

/**
 * Peeks 2 bytes form head of buffered stream.
 *
 * @param buffer
 */
uint16_t peek_marker(type_buffer *buffer) {
	if(buffer->bp+1 >= buffer->end)
	{
		println_var(INFO, "Error: Exceeded buffer bounds!");
		exit(0);
	}
	uint16_t val = (*buffer->bp)<<8;
	val += *(buffer->bp+1);
	return val;
}

void skip_buffer(type_buffer *buffer, int n)
{
	buffer->bp += n;
}

int32_t tell_buffer(type_buffer *buffer)
{
	return buffer->bp - buffer->start;
}

uint32_t read_byte_(type_buffer *buffer)
{
	buffer->byte = (buffer->byte << 8) & 0xffff;
	buffer->bits_count = buffer->byte == 0xff00 ? 7 : 8;
	if(buffer->bp >= buffer->end)
	{
		return 1;
	}
	buffer->byte |= *buffer->bp++;
	return 0;
}

uint32_t read_bit(type_buffer *buffer)
{
	if(buffer->bits_count == 0)
	{
		read_byte_(buffer);
	}
	buffer->bits_count--;
	return (buffer->byte >> buffer->bits_count) & 1;
}

uint32_t read_bits(type_buffer *buffer, int n)
{
	int i, val = 0;

	for(i = n -1; i >= 0; i--)
	{
		val += read_bit(buffer) << i;
	}

	return val;
}

uint32_t inalign(type_buffer *buffer)
{
	buffer->bits_count = 0;
	if((buffer->byte & 0xff) == 0xff)
	{
		if(read_byte_(buffer))
		{
			return 1;
		}
		buffer->bits_count = 0;
	}
	return 0;
}

void enlarge_buffer_n(type_buffer *buffer, int size)
{
	uint8_t *old_data = buffer->data;
	/* Enlarge buffer to new_size */
	uint32_t new_size = size;
	buffer->data = (uint8_t*)malloc(new_size);
	memcpy(buffer->data, old_data, buffer->size);
	buffer->size = new_size;
	free(old_data);
}

/**
 * Enlarges buffer by the power of 2 if there is insufficient memory space.
 *
 * @param buffer
 */
void enlarge_buffer(type_buffer *buffer)
{
	enlarge_buffer_n(buffer, buffer->size * 2);
}

/**
 * Writes byte to buffer.
 *
 * @param buffer
 * @param val
 */
void write_byte(type_buffer *buffer, uint8_t val)
{
	if(buffer->bytes_count == buffer->size)
	{
		enlarge_buffer(buffer);
	}

	buffer->data[buffer->bytes_count] = val;
//	printf("(%u,%x)\n", val, val);
	buffer->bytes_count++;
}

/**
 * Writes short to buffer.
 *
 * @param buffer
 * @param val
 */
void write_short(type_buffer *buffer, uint16_t val)
{
	write_byte(buffer, val >> 8);
	write_byte(buffer, val);
}

/**
 * Writes int to buffer.
 *
 * @param buffer
 * @param val
 */
void write_int(type_buffer *buffer, uint32_t val)
{
	write_byte(buffer, val >> 24);
	write_byte(buffer, val >> 16);
	write_byte(buffer, val >> 8);
	write_byte(buffer, val);
}

/**
 * @brief If the value of the byte is 0xFF, the next byte includes an extra zero bit stuffed into the MSB.
 *
 * @param buffer
 */
void bit_stuffing(type_buffer *buffer)
{
	if(buffer->bits_count == 8)
	{
		buffer->bits_count = 0;
		write_byte(buffer, buffer->byte);
		if(buffer->byte == 0xFF)
		{
			buffer->bits_count = 1;
		}
		buffer->byte = 0;
	}
}

void write_stuffed_byte(type_buffer *buffer)
{
	if(buffer->bits_count == 8)
	{
		printf("Error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
	}

	if(buffer->bits_count != 0)
	{
		buffer->bits_count = 0;
		write_byte(buffer, buffer->byte);
		buffer->byte = 0;
	}

//	bit_stuffing(buffer);
//	write_stuffed_byte(buffer);
}

void write_zero_bit(type_buffer *buffer)
{
	buffer->bits_count++;
	bit_stuffing(buffer);
}

void write_one_bit(type_buffer *buffer)
{
	buffer->byte |= 1 << (7 - buffer->bits_count);
	buffer->bits_count++;
	bit_stuffing(buffer);
}

void write_bits(type_buffer *buffer, int bits, int n)
{
	int bit = 0;
	int i;
	for(i = n - 1; i >= 0; i--)
	{
		bit = bits & (1 << i);
		if(bit == 0)
		{
			write_zero_bit(buffer);
//			printf("0");
		} else
		{
			write_one_bit(buffer);
//			printf("1");
		}
	}
}

void write_array(type_buffer *buffer, uint8_t *in, int length)
{
	int i;

	if(buffer->bits_count != 0)
	{
		println_var(INFO, "ERROR! bits_count sholud be equal 0");
	}

	if(buffer->bytes_count + length >= buffer->size)
	{
//		println_var(INFO, "buffer->size:%d", buffer->size);
		int new_size = buffer->size;
		while(buffer->bytes_count + length >= new_size)
		{
			new_size *= 2;
		}
		enlarge_buffer_n(buffer, new_size);
//		println_var(INFO, "new buffer->size:%d", buffer->size);
	}

	memcpy(buffer->data + buffer->bytes_count, in, length);
	buffer->bytes_count += length;

//	for(i = 0; i < length; i++)
//	{
//		write_byte(buffer, in[i]);
//	}
}

void update_buffer_byte(type_buffer *buffer, int pos, uint8_t val)
{
	buffer->data[pos] = val;
}

/**
 * Writes buffer to file.
 *
 * @param buffer
 * @param fp
 */
void write_buffer_to_file(type_buffer *buffer, FILE *fp)
{
	fwrite(buffer->data, sizeof(uint8_t), buffer->bytes_count, fp);
}
