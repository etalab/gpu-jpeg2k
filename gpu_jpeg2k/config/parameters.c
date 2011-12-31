/**
 * @file parameters.c
 * @brief Configuration file processed here. The configuration values supplied by the user
 *  	  in the configuration file are stored in variables defined in parameters.h
 *
 *
 * @author: Jakub Misiorny <misiorny@man.poznan.pl>
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "parameters.h"
#include "ini.h"
#include "../print_info/print_info.h"

void default_config_values(type_parameters *param) {
	param->param_tile_w = -1;
	param->param_tile_h = -1;
	param->param_tile_comp_dlvls = 4;
	param->param_cblk_exp_w = 6;
	param->param_cblk_exp_h = 6;
	param->param_wavelet_type = 1;
	param->param_use_mct = 0;
	param->param_device = 0;
	param->param_target_size = 0;
	param->param_bp = 0.0;
	param->param_use_part2_mct = 0;
	param->param_mct_compression_method = 0;
	param->param_mct_klt_iterations = 10000;
	param->param_mct_klt_err = 1.0e-7;
	param->param_mct_klt_border_eigenvalue = 0.0001;
}

static int handler(/*@unused@*/void* user, /*@unused@*/const char* section, const char* name, const char* value,
		type_parameters *param) {
#define MATCH(n) strcmp(name, n) == 0

	if (MATCH("tile_w")) {
		param->param_tile_w = atoi(value);
	} else if (MATCH("tile_h")) {
		param->param_tile_h = atoi(value);
	} else if (MATCH("tile_comp_dlvls")) {
		param->param_tile_comp_dlvls = atoi(value);
	} else if (MATCH("cblk_exp_w")) {
		param->param_cblk_exp_w = atoi(value);
	} else if (MATCH("cblk_exp_h")) {
		param->param_cblk_exp_h = atoi(value);
	} else if (MATCH("wavelet_type")) {
		param->param_wavelet_type = atoi(value);
	} else if (MATCH("use_mct")) {
		param->param_use_mct = atoi(value);
	} else if (MATCH("device")) {
		param->param_device = atoi(value);
	} else if (MATCH("target_size")) {
		param->param_target_size = atoi(value);
	} else if (MATCH("bp")) {
		param->param_bp = (float) atof(value);
	} else if (MATCH("use_part2_mct")) {
		param->param_use_part2_mct = atoi(value);
	} else if (MATCH("mct_compression_method")) {
		param->param_mct_compression_method = atoi(value);
	} else if (MATCH("mct_klt_iterations")) {
		param->param_mct_klt_iterations = atoi(value);
	} else if (MATCH("mct_klt_err")) {
		param->param_mct_klt_err = (float) atof(value);
	} else if (MATCH("mct_klt_border_eigenvalue")) {
		param->param_mct_klt_border_eigenvalue = (float) atof(value);
	}

	//1 is success
	return 1;
}

int parse_config(const char *filename, type_parameters *param) {

	default_config_values(param);

	//	println_var(INFO, "Loading: %s", filename);
	if (filename != NULL) {
		if (ini_parse(filename, handler, "", param) < 0) {
			fprintf(stdout, "Can't load '%s. Using default config.\n", filename);
			return 0;
		}
	}

	//    println_var(INFO, "Configuration parsed sucessfully");
	return 0;
}
