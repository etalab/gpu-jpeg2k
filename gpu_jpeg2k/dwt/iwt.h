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
