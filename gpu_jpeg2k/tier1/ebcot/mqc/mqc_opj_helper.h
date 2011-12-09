#ifndef _MQC_OPJ_HELPER_H
#define _MQC_OPJ_HELPER_H

#include "library/openjpeg/openjpeg.h"

/**
 * Reset openjpeg library
 *
 * @return void
 */
void
mqc_opj_helper_reset();

void
mqc_opj_helper_parameters(unsigned int cblk_width, unsigned int cblk_height, int dwt, bool irreversible);

/**
 * Encode file with openjpeg library
 *
 * @param filename Filename of bmp image
 * @param callback_image_info callback to get image info
 * @param param param for callback
 * @return true if succeeded
 */
bool
mqc_opj_helper_encode(const char* filename, void(*callback_image_info)(opj_image_t*,void*), void* param);

/**
 * Encode file with openjpeg library with duration
 *
 * @param filename Filename of bmp image
 * @param callback_image_info callback to get image info
 * @param param param for callback
 * @param duration
 * @return true if succeeded
 */
bool
mqc_opj_helper_encode_with_duration(const char* filename, void(*callback_image_info)(opj_image_t*,void*), void* param, double* duration);

#endif
