/**
 * @file iwt_1d.h
 *
 * @author Milosz Ciznicki
 */

#ifndef IWT_1D_H_
#define IWT_1D_H_

#include "../types/image_types.h"

#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif

void iwt_1d(type_image *img, int lvl);

#endif /* IWT_1D_H_ */
