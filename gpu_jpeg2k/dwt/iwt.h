/**
 * @file iwt.h
 *
 * @author Milosz Ciznicki
 */


#ifndef IWT_H_
#define IWT_H_

#include "dwt.h"

extern __global__
void iwt97(const float *idata, float *odata, const int2 img_size, const int2 step);

extern __global__
void iwt53(const float *idata, float *odata, const int2 img_size, const int2 step);

#endif /* IWT_H_ */
