/**
 * @file image_bip.h
 *
 * @author Milosz Ciznicki
 */

#ifndef IMAGE_BIP_H_
#define IMAGE_BIP_H_

#include <stdio.h>
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include "../types/image_types.h"
#include "../config/parameters.h"
#include "image_hyper.h"

#define DEPTH 16

typedef struct type_image_hyper type_image_bip;

type_data *read_bip_image(type_image *container, type_image_hyper *image_bip, type_parameters *param);
void save_raw(type_image *img, const char* out_file);

#endif /* IMAGE_BIP_H_ */
