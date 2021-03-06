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
 * @file image_bsq.h
 *
 * @author Milosz Ciznicki
 */

#ifndef IMAGE_BSQ_H_
#define IMAGE_BSQ_H_

#include <stdio.h>
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include "../types/image_types.h"
#include "../config/parameters.h"
#include "image_hyper.h"

#define DEPTH 16

type_data *read_bsq_image(type_image *_container, type_image_hyper *image_bsq, type_parameters *param);
void save_raw(type_image *img, const char* out_file);

#endif /* IMAGE_BSQ_H_ */
