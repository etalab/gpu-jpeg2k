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
