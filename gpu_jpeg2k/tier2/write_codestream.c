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
/*
 * write_codestream.c
 *
 *  Created on: Dec 1, 2011
 *      Author: miloszc
 */
#include <stdlib.h>
#include "../my_common/my_common.h"
#include <stdio.h>

#include "write_codestream.h"

void write_codestream(type_image *img) {
	type_buffer *buffer = (type_buffer *) my_malloc(sizeof(type_buffer));

	init_buffer(buffer);
	encode_codestream(buffer, img);

	FILE *fp = fopen(img->out_file, "wb");

	write_buffer_to_file(buffer, fp);

	fclose(fp);
	free(buffer);
}
