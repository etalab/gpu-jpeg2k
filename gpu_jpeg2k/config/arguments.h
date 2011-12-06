/*
 * arguments.h
 *
 *  Created on: Nov 22, 2011
 *      Author: Milosz Ciznicki
 */

#ifndef ARGUMENTS_H_
#define ARGUMENTS_H_

#include "../types/image_types.h"
#include "parameters.h"

#define ERROR -1

int parse_args(int argc, char **argv, type_image *img);
int check_args_enc(type_image *img);
int check_args_dec(type_image *img);
int check_config(type_parameters *param);

#endif /* ARGUMENTS_H_ */
