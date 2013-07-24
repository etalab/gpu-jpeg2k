#ifndef KLT_H_
#define KLT_H_

#include "../types/image_types.h"
#include "../config/parameters.h"

void encode_klt(type_parameters *param, type_image *img);
void decode_klt(type_image *img);

#endif
