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
 * @brief Constants for the lossy color transformation.
 *
 * source: http://en.wikipedia.org/wiki/YUV#Conversion_to.2Ffrom_RGB
 *
 * @file preprocessing_constants.cuh
 *
 * @author Jakub Misiorny <misiorny@man.poznan.pl>
 */

#ifndef PREPROCESSING_CONSTANTS_CUH_
#define PREPROCESSING_CONSTANTS_CUH_

const float Wr = 0.299f;
const float Wb = 0.114f;
//const float Wg = 1 - Wr - Wb;
const float Wg = 1.0f - 0.299f - 0.114f;
const float Umax = 0.436f;
const float Vmax = 0.615f;



#endif /* PREPROCESSING_CONSTANTS_CUH_ */
