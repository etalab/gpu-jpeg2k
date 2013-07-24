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
 * gpu_convert.h
 *
 *  Created on: Feb 20, 2012
 *      Author: miloszc
 */

#ifndef GPU_CONVERT_H_
#define GPU_CONVERT_H_

void convert(dim3 gridDim, dim3 blockDim, CodeBlockAdditionalInfo *infos, unsigned int *g_icxds, unsigned char *g_ocxds, const int maxOutLength);


#endif /* GPU_CONVERT_H_ */
