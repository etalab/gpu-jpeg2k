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
