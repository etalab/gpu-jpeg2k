/**
 * @file iwt.h
 *
 * @author Milosz Ciznicki
 */


#ifndef IWT_NEW_H_
#define IWT_NEW_H_

#include "dwt.h"

extern __global__
void iwt97_new(const float *idata, float *odata, const int2 img_size, const int2 step);

extern __global__
void iwt53_new(const float *idata, float *odata, const int2 img_size, const int2 step);

#endif /* IWT_NEW_H_ */
