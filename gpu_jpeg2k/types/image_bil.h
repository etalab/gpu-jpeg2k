/**
 * @file image_bil.h
 *
 * @author Milosz Ciznicki
 */

#ifndef image_BIL_H_
#define image_BIL_H_

#include <stdio.h>
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include "../types/image_types.h"
#include "../config/parameters.h"
#include "image_hyper.h"

#define DEPTH 16

typedef struct type_image_hyper type_image_bil;

type_data *read_bil_image(type_image *_container, type_image_hyper *image_bil, type_parameters *param);
void save_raw(type_image *img, const char* out_file);

#endif /* image_BIL_H_ */
