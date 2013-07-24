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
 * mean_adjust_data.h
 *
 *  Created on: Nov 30, 2011
 *      Author: miloszc
 */

#ifndef MEAN_ADJUST_DATA_H_
#define MEAN_ADJUST_DATA_H_

void mean_adjust_data(type_image *img, type_data** data, type_data* means);
void mean_adjust_data_new(type_image *img, type_data** data, type_data* means);

#endif /* MEAN_ADJUST_DATA_H_ */
