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
 * reduce.h
 *
 *  Created on: Nov 30, 2011
 *      Author: miloszc
 */

#ifndef REDUCE_H_
#define REDUCE_H_

#include "../types/image_types.h"

#define MAX_BLOCKS 65535

type_data reduction(type_data *d_i_data, type_data *d_o_data, type_data *h_odata, int size);

#endif /* REDUCE_H_ */
