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
/*
 * shift.h
 *
 *  Created on: Nov 30, 2011
 *      Author: miloszc
 */

#ifndef SHIFT_H_
#define SHIFT_H_

#include <stdint.h>
#include "../types/image_types.h"

void shit(type_data *idata, const uint16_t w, const uint16_t h, const type_data mean);

#endif /* SHIFT_H_ */
