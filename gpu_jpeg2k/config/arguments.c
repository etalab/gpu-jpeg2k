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
 * arguments.c
 *
 *  Created on: Nov 22, 2011
 *      Author: Milosz Ciznicki
 */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "../my_common/my_common.h"
#include <unistd.h>
#include "arguments.h"
#include "../types/image_types.h"

#define N_REQ_OPTS 2
#define REQ_OPTS 1 << N_REQ_OPTS

int parse_args(int argc, char **argv, type_image *img) {
	uint8_t flag = 1;
	char c;
	while ((c = getopt(argc, argv, "i:o:h:c:")) != -1)
		switch (c) {
		case 'i':
			img->in_file = optarg;
			flag <<= 1;
			break;
		case 'h':
			img->in_hfile = optarg;
			break;
		case 'o':
			img->out_file = optarg;
			flag <<= 1;
			break;
		case 'c':
			img->conf_file = optarg;
			break;
		case '?': {
			switch (optopt) {
			case 'i':
				fprintf(stderr, "Option -%i requires an input file.\n", optopt);
				break;
			case 'h':
				fprintf(stderr, "Option -%i requires an header file.\n", optopt);
				break;
			case 'o':
				fprintf(stderr, "Option -%i requires an output file.\n", optopt);
				break;
			case 'c':
				fprintf(stderr, "Option -%i requires an configuration file.\n", optopt);
				break;
			default:
				fprintf(stderr, "Unknown option -%i.\n", optopt);
				break;
			}
			break;
		}
		case ':':
			fprintf(stderr, "Unknown option -%i.\n", optopt);
			break;
		default:
			return ERROR;
		}

	int i = 0;
	for (i = optind; i < argc; ++i) {
		printf("Non-option argument %s\n", argv[i]);
	}

	if (optind < argc || (flag != (REQ_OPTS)))
		return ERROR;
	else
		return 0;
}

static int check_args(type_image *img) {
	if (img->conf_file != NULL) {
		if (strstr(img->conf_file, ".config") == NULL) {
			fprintf(stderr, "Wrong extension: %s (Should be *.config)\n", img->conf_file);
			return ERROR;
		}
	}
	return 0;
}

int check_args_enc(type_image *img) {
	if (strstr(img->in_file, ".bsq") != NULL)
		img->bsq_file = 1;
	if (strstr(img->in_file, ".bip") != NULL)
		img->bip_file = 1;
	if (strstr(img->in_file, ".bil") != NULL)
		img->bil_file = 1;

	if (img->in_hfile != NULL) {
		if (strstr(img->in_hfile, ".hdr") == NULL) {
			fprintf(stderr, "Wrong header file: %s! (Should be *.hdr)\n", img->in_hfile);
			return ERROR;
		} else {
			if ((strstr(img->in_file, ".bsq") == NULL) && (strstr(img->in_file, ".bip") == NULL) && (strstr(
					img->in_file, ".bil") == NULL)) {
				fprintf(stderr, "Wrong image file: %s! (Should be one of following: *.bsq, *.bip, *.bil)\n",
						img->in_file);
				return ERROR;
			}
		}
	} else {
		if (strstr(img->out_file, ".j2k") == NULL) {
			fprintf(stderr, "Wrong extension: %s (Should be *.j2k)\n", img->out_file);
			return ERROR;
		}
	}

	return check_args(img);
}

int check_args_dec(type_image *img) {
	if (strstr(img->out_file, ".bsq") != NULL)
		img->bsq_file = 1;
	if (strstr(img->out_file, ".bip") != NULL)
		img->bip_file = 1;
	if (strstr(img->out_file, ".bil") != NULL)
		img->bil_file = 1;


	if (img->in_file != NULL) {
		if ((strstr(img->in_file, ".j2k") == NULL) && (strstr(img->in_file, ".jp2") == NULL)) {
			fprintf(stderr, "Wrong input file: %s! (Should be *.j2k or *.jp2)\n", img->in_hfile);
			return ERROR;
		}
	}
	return check_args(img);
}

int check_config(type_parameters *param) {
	if ((param->param_use_mct) == 1 && (param->param_use_part2_mct == 1)) {
		fprintf(stderr, "Conflicting options: use_mct and use_art2_mct\n");
		return ERROR;
	}
	if ((param->param_wavelet_type < 0) || (param->param_wavelet_type > 1)) {
		fprintf(stderr, "Wrong wavelet_type: %d (Should be 0 (lossles) or 1 (lossy))\n", param->param_wavelet_type);
		return ERROR;
	}
	return 0;
}
