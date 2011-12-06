/**
 * @file encoder.c
 * @brief This is the main file for the encoder component. It executes all modules in order producing JPEG 2000 file/stream.
 *
 * @author Jakub Misiorny <misiorny@man.poznan.pl>
 * @author Milosz Ciznicki
 * @date Nov 16, 2010
 */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include "print_info/print_info.h"

#include "preprocessing/preprocess_gpu.h"
#include "preprocessing/mct.h"
#include "config/parameters.h"
#include "config/arguments.h"
#include "config/help.h"

#include "types/image.h"
#include "types/image_types.h"
#include "types/buffered_stream.h"

#include "dwt/dwt.h"
#include "dwt/fwt_1d.h"
#include "dwt/iwt_1d.h"

#include "tier1/quantizer.h"
#include "tier1/coeff_coder/gpu_coder.h"

#include "tier2/codestream.h"
#include "tier2/write_codestream.h"

#define GLOBAL_TIME
//#define PART_TIME

/**
 * @brief Main program entrance.
 * Input parameters:
 * 1) -i in_image
 * 2) -o out_image
 * 3) [-h input header file]
 * 4) [-c configuration file]
 *
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv)
{
//	println_start(INFO);
	type_image *img = (type_image *)malloc(sizeof(type_image));
	memset(img, 0, sizeof(type_image));
	if((parse_args(argc, argv, img) == ERROR) || (check_args_enc(img) == ERROR))
	{
		fprintf(stderr, "Error occurred while parsing arguments.\n");
		fprintf(stdout, "%s", help);
		return 1;
	}

	type_parameters *param = (type_parameters*)malloc(sizeof(type_parameters));
	if((parse_config(img->conf_file, param) == ERROR) || (check_config(param) == ERROR)) {
		fprintf(stderr, "Error occurred while parsing configuration file.\n");
		fprintf(stdout, "%s", help);
		return 1;
	}

	init_device(param);

#ifdef PART_TIME
	long int start_read;
	start_read = start_measure();
#endif

	if(read_image(img, param) == ERROR) {
		fprintf(stderr, "Can not read image.\n");
		return 1;
	}

#ifdef PART_TIME
	cudaThreadSynchronize();
//	printf("Read img:%ld\n", stop_measure(start_read));
#endif

#ifdef GLOBAL_TIME
	long int start_global;
	start_global = start_measure();
#endif

#ifdef PART_TIME
	long int start_mct;
	start_mct = start_measure();
#endif

	mct(img, param);

#ifdef PART_TIME
	cudaThreadSynchronize();
	printf("%ld\n", stop_measure(start_mct));
#endif
	int i = 0;
	type_tile *tile = NULL;
	// Do processing for all tiles
	for(i = 0; i < img->num_tiles; i++)
	{
		tile = &(img->tile[i]);

#ifdef PART_TIME
	long int start_dwt;
	start_dwt = start_measure();
#endif
		// Do forward wavelet transform
		fwt(tile);

#ifdef PART_TIME
	cudaThreadSynchronize();
	printf("%ld\n", stop_measure(start_dwt));
#endif

#ifdef PART_TIME
	long int start_qs;
	start_qs = start_measure();
#endif
		// Divide subbands to codeblocks, than quantize data
		quantize_tile(tile);

#ifdef PART_TIME
	cudaThreadSynchronize();
	printf("%ld\n", stop_measure(start_qs));
#endif

#ifdef PART_TIME
	long int start_coder;
	start_coder = start_measure();
#endif

		encode_tile(tile, 0);

#ifdef PART_TIME
	cudaThreadSynchronize();
	printf("%ld\n", stop_measure(start_coder));
#endif
	}

#ifdef PART_TIME
	long int start_cd;
	start_cd = start_measure();
#endif
	/* Write Codestream */
	write_codestream(img);

#ifdef PART_TIME
	cudaThreadSynchronize();
	printf("%ld\n", stop_measure(start_cd));
#endif

#ifdef GLOBAL_TIME
	printf("%ld\n", stop_measure(start_global)/* + copy_time*/);
#endif

	free(img);

//	println_end(INFO);
	return 0;
}
