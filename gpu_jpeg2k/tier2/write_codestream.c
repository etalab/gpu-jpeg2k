/*
 * write_codestream.c
 *
 *  Created on: Dec 1, 2011
 *      Author: miloszc
 */
#include <stdlib.h>
#include <stdio.h>

#include "write_codestream.h"

void write_codestream(type_image *img) {
	type_buffer *buffer = (type_buffer *) malloc(sizeof(type_buffer));

	init_buffer(buffer);
	encode_codestream(buffer, img);

	FILE *fp = fopen(img->out_file, "wb");

	write_buffer_to_file(buffer, fp);

	fclose(fp);
	free(buffer);
}
