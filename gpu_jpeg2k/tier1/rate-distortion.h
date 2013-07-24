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
 * @file rate-distortion.h
 *
 * @author milosz 
 * @date 29-09-2010
 */

#ifndef RATE_DISTORTION_H_
#define RATE_DISTORTION_H_

#include "mq_coder.h"

void near_optimal_truncation_length(mqe_t *mq, int rates[], int last_byte, int out_buff[], int n);
void compute_convex_hull(int rates[], float dists[], int n);

#endif /* RATE_DISTORTION_H_ */
