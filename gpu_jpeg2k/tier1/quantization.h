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
 * @file quantization.h
 *
 * @author Milosz Ciznicki
 */

#ifndef QUANTIZATION_H_
#define QUANTIZATION_H_

#define BLOCKSIZEX 16
#define BLOCKSIZEY 16
#define COMPUTED_ELEMS_BY_THREAD 4

float convert_from_exp_mantissa(int ems);
int convert_to_exp_mantissa(float step);
int get_exp_subband_gain(int orient);

#endif /* QUANTIZATION_H_ */
