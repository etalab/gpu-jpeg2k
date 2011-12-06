/**
 * 	@file decoder.c
 *	@brief This the JP2 fileformat decoder main file. It loads JPEG 2000 file format.
 *
 *	Implementation of Appendix I from the standard. To be merged with mainstream decoder.c
 *
 *  @author Jakub Misiorny <misiorny@man.poznan.pl>
 */


#include <stdlib.h>
#include <stdio.h>

#include "types/image_types.h"
#include "config/parameters.h"

#include "print_info/print_info.h"

#include "file_format/boxes.h"


int main(int argc, char **argv)
{
	println_start(INFO);

	int i;
	type_parameters *param = (type_parameters*)malloc(sizeof(type_parameters));
	type_tile *tile;

	if (argc < 2)
	{
		printf("Usage: decoder <image-file-name>\n\7");
		exit(0);
	}

	default_config_values(param);

	println_var(INFO, "%s %s", argv[0], argv[1]);

	type_image *img = (type_image *)malloc(sizeof(type_image));

	FILE *fsrc = fopen(argv[1], "rb");
	if (!fsrc) {
		println_var(INFO, "[ERROR] failed to open %s for reading\n", argv[1]);
		return 1;
	}
	fseek(fsrc, 0, SEEK_END);
	long file_length = ftell(fsrc);
	fseek(fsrc, 0, SEEK_SET);


	println_var(INFO, "Loaded file length: %li", file_length);

	jp2_parse_boxes(fsrc, img);

	return 0;
}
