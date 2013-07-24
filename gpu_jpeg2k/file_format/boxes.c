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
 * @file boxes.c
 * @brief Parsing of boxes
 *
 * @author Jakub Misiorny <misiorny@man.poznan.pl>
 */
#include <stdlib.h>
#include <string.h>

#include "boxes.h"
#include "../print_info/print_info.h"
#include "../types/buffered_stream.h"

#include "../tier2/codestream.h"

box *init_box() {
	box *new = malloc(sizeof(box));
	new->lbox = (unsigned char *)malloc(sizeof(char) * 5);
	new->tbox = (unsigned char *)malloc(sizeof(char) * 5);
}

//dest has to be n+1 long
char *sstrncpy(char *dest, const char *src, size_t n) {
	println_start(INFO);

	size_t i;

	for (i = 0; i < n; i++) {
//		printf("i: %i\n", i);
		dest[i] = src[i];
	}

	println(INFO, "middle");
	for (; i <= n; i++)
		dest[n] = '\0';


	return dest;
}


long int hex_to_long(unsigned char *hex, int length) {
	long int ret = hex[length-1];
	long tmp = 1;
	int i;
//	printf("1 * %x\n", hex[length-1]);
	for(i = 2; i <= length; i++) {
		tmp *= 256;
//		printf("%li * %x\n", tmp, hex[length-i]);
		ret += tmp*hex[length-i];
	}
	return ret;
}

unsigned char *read_bytes(unsigned char *dest, FILE *fd, long int n) {
	long int i;
	for(i = 0; i< n; i++) {
		int tmp = fgetc(fd);

		if(tmp == EOF) {
			println(INFO, "EOF");
			return NULL;
		}

		dest[i] = (unsigned char)tmp;
	}

	dest[i] = '\0';
	return (unsigned char *)dest;
}


//box *get_next_box(unsigned char *mem, long int *pos) {
box *get_next_box(FILE *fd) {
	println_start(INFO);
	//box *box = malloc(sizeof(box));

	int read = 0;
	box *box = init_box();

	if( (box->lbox = read_bytes(box->lbox, fd, 4)) == NULL)
		return NULL;

	box->length = hex_to_long(box->lbox, 4);
	if(box->length == 0) {
		return NULL;
	}

	if( (box->tbox = read_bytes(box->tbox, fd, 4)) == NULL) {
		println(INFO, "Corrupted JP2 file. Exitting.");
		return NULL;
	}
	read = 8;

	if(box->length == 1) { //there should be XLbox field present;
		box->xlbox = malloc(9 * sizeof(char));
		box->xlbox = read_bytes(box->xlbox, fd, 8);
		box->length = hex_to_long(box->xlbox, 8);
		read += 8;
	}
	box->content_length = box->length - read;

	box->dbox = malloc((box->content_length + 1) * sizeof(char));
	box->dbox = read_bytes(box->dbox, fd, box->content_length);
	println_end(INFO);
	return box;
}

box *get_next_box_char(box *superbox) {
	if(superbox->read >= superbox->content_length)
		return NULL;

	char *content = superbox->dbox;
	box *box = malloc(sizeof(box));

	int read = superbox->read;
	box->lbox = malloc(5 * sizeof(char));
	box->lbox = sstrncpy(box->lbox, &content[read], 4); read += 4;
	box->length = hex_to_long(box->lbox, 4);

	println_var(INFO, "lbox: %i", box->length);
	box->tbox = malloc(5 * sizeof(char));
	box->tbox = sstrncpy(box->tbox, &content[read], 4); read += 4;

	println(INFO, "tbox");
	//box->length = hex_to_long(box->lbox, 4);
//	println_var(INFO, "length: %i", box->length);
	if(box->length == 1) { //there should be XLbox field present;
		box->xlbox = malloc(9 * sizeof(char));
		box->xlbox = sstrncpy(box->xlbox, &content[read], 8);
		box->length = hex_to_long(box->xlbox, 8);
		read += 8;
	}

	int size = (box->length - (read - superbox->read));
	box->dbox = malloc((size+1) * sizeof(char));
	box->dbox = sstrncpy(box->dbox, &content[read], size);
	read += size;

	box->content_length = box->length - read;
	println(INFO, "dbox");
	superbox->read = read;
	println_end(INFO);
	return box;
}

int dispose_of(box *b) {
/*	println_start(INFO);
	if(b->lbox == NULL)
		printf("NULL!\n");
	else
		free(b->lbox);
	println(INFO, "A");

	free(b->tbox);
	println(INFO, "A");
	free(b->dbox);

	println(INFO, "A");
	if(b->xlbox != NULL)
		free(b->xlbox);

	println(INFO, "A");
*/
	free(b);
	println_end(INFO);
	return 0;
}

int h_filetype_box(box *b, type_image *img) {
	char *br = malloc(5 * sizeof(char));
	br = strncpy(br, b->dbox, 4);

	if(strcmp(br, "jp2\040")) {
		println(INFO, "DOSEN'T Conform to IS 15444-1. Exitting");
		return 1;
	} else
		println(INFO, "Conforms to IS 15444-1");

	char *minv = malloc(5 * sizeof(char));
	minv = strncpy(minv, &(b->dbox[4]), 4);

	if(hex_to_long(minv, 4) != 0) {
		println(INFO, "MinV should be 0");
	} else
		println(INFO, "MinV OK");

	int left = b->content_length - 8; //16 bytes already read from the box's contents: br,minv
	printf("left: %i\n", left);
	char *cl;
	int i = 1;
	while(left) {
		cl = malloc(5 * sizeof(char));
		cl = strncpy(cl, &(b->dbox)[4 + i*4], 4); left -= 4; i++;

		//TODO: filetype box: codestream profile restrictions
		if(!strcmp(cl, "J2P0")) {
			println(INFO, "First codestream is restricted: Profile-0");
		} else
		if(!strcmp(cl, "J2P1")) {
			println(INFO, "First codestream is restricted: Profile-1");
		} else
		if(!strcmp(cl, "jp2\040")) {
			println(INFO, "One CL field with correct fingerprint");
		} else
			println_var(INFO, "CL: >%s<", cl);
	}
	return 0;
}

int h_image_header_box(box *b, type_image *img) {
	char *cheight = malloc(5 * sizeof(char));
	cheight = strncpy(cheight, b->dbox, 4);
	img->height = hex_to_long(cheight, 4);

	char *cwidth = malloc(5 * sizeof(char));
	cwidth = strncpy(cwidth, &(b->dbox)[4], 4);
	img->width = hex_to_long(cwidth, 4);

	char *cnum_comp = malloc(3 * sizeof(char));
	cnum_comp = strncpy(cnum_comp, &(b->dbox)[8], 2);
	img->num_components = hex_to_long(cnum_comp, 2);

	//TODO: bpc

	if(b->dbox[11] != 7) {
		println_var(INFO, "Value of the 'Compression type' shall be 7, not: %i", b->dbox[11]);
	}

	if(b->dbox[12] ==  0) {
		println(INFO, "Colorspace know. Specified in Colorspace box");
	} else
	if(b->dbox[12] == 1) {
		println(INFO, "Colorspace UNKNOWN.");
	}
}

int h_header_box(box *header_box, type_image *img) {
	println_start(INFO);
	header_box->read = 0;
	box *ihdr = get_next_box_char(header_box); //TODO: extract box from within b

	if(hex_to_long(ihdr->tbox, 4) != IMAGE_HEADER_BOX) {
		println(INFO, "Image Header Box should be the first one in Header superbox. Exitting!");
		return 1;
	} else
		h_image_header_box(ihdr, img);

	box *b;
	while( (b = get_next_box_char(header_box)) != NULL ) {
			if(hex_to_long(b->tbox, 4) == BITS_PER_COMPONENT_BOX) {
				println(INFO, "Bits Per Component box");
			} else
			if(hex_to_long(b->tbox,4) ==  COLOR_BOX) {
				println(INFO, "Color Specification Box");
			} else
			if(hex_to_long(b->tbox,4) ==  PALETTE_BOX) {
				println(INFO, "Palette Box");
			} else
			if(hex_to_long(b->tbox,4) ==  COMPONENT_MAPPING_BOX) {
				println(INFO, "Component Mapping Box");
			} else
			if(hex_to_long(b->tbox,4) ==  CHANNEL_DEFINITION_BOX) {
				println(INFO, "Channel Definition Box");
			}
	}

	return 0;
}

h_contiguous_codestream_box(box *cbox, type_image *img) {

	long int codestream_len = cbox->content_length;

	type_buffer *src_buff = (type_buffer *) malloc(sizeof(type_buffer));
	src_buff->data = (uint8_t *) malloc(codestream_len+1);
	src_buff->size = codestream_len;

	src_buff->data = cbox->dbox;

	//src_buff->data = sstrncpy(src_buff->data, cbox->dbox, codestream_len);

	src_buff->start = src_buff->data;
	src_buff->end = src_buff->data + src_buff->size;
	src_buff->bp = src_buff->data;
	src_buff->bits_count = 0;
	src_buff->byte = 0;

	println(INFO, "Decoding codestream");
	decode_codestream(src_buff, img);

	println_end(INFO);
}

int jp2_parse_boxes(FILE *fd, type_image *img) {
	box *sig = get_next_box(fd);

	if(hex_to_long(sig->tbox, 4) != JP2_SIGNATURE_BOX)
		println(INFO, "JP2 signature box should be very first box in the file: Header");

	if(hex_to_long(sig->dbox, 4) != JP2_SIG_BOX_CONTENT)
		println(INFO, "JP2 signature box should be very first box in the file: Content");

	//dispose_of(sig);
	//free(sig);

	box *ft = get_next_box(fd);

	if(hex_to_long(ft->tbox, 4) != JP2_FILETYPE_BOX)
		println(INFO, "JP2 filetype box should directly follow JP2 signature box");

	if(h_filetype_box(ft, img))
		return 1;

	box *b;
	while( (b = get_next_box(fd)) != NULL ) {
		if(hex_to_long(b->tbox, 4) == JP2_HEADER_BOX) {
			println(INFO, "Header Box");
			h_header_box(b,img);
		} else
		if(hex_to_long(b->tbox,4) ==  CODE_STREAM_BOX) {
			println(INFO, "Contiguous Codestream Box");
			h_contiguous_codestream_box(b,img);

		} else
		if(hex_to_long(b->tbox,4) ==  INTELLECTUAL_PROPERTY_BOX) {
			println(INFO, "Intellectual Property Box");
		}
	}

	println_end(INFO);
	return 0;
}
