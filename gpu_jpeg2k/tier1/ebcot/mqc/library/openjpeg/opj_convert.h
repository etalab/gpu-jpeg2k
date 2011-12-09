#ifndef __OPJ_CONVERT_H
#define __OPJ_CONVERT_H

#include "openjpeg.h"

opj_image_t*
opj_bmp_to_image(const char *filename, opj_cparameters_t *parameters);

opj_image_t*
opj_pgx_to_image(const char *filename, opj_cparameters_t *parameters);

opj_image_t*
opj_pnm_to_image(const char *filename, opj_cparameters_t *parameters);

typedef struct raw_cparameters {
	/** width of the raw image */
	int rawWidth;
	/** height of the raw image */
	int rawHeight;
	/** components of the raw image */
	int rawComp;
	/** bit depth of the raw image */
	int rawBitDepth;
	/** signed/unsigned raw image */
	bool rawSigned;
} raw_cparameters_t;

opj_image_t*
opj_raw_to_image(const char *filename, opj_cparameters_t *parameters, raw_cparameters_t *raw_cp);

#endif

