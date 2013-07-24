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
 * nl_blocks.h
 *
 *  Created on: 2010-05-16
 *      Author: Milosz Ciznicki
 */

#ifndef BLOCKS_H_
#define BLOCKS_H_

/*
 * @brief Get next power of two.
 */
unsigned int next_pow2(unsigned int x);

/*
 * @brief Compute the number of threads and blocks to use for the given reduction kernel
 * For the kernels >= 3, we set threads / block to the minimum of maxThreads and
 * n/2. For kernel
 * 6, we observe the maximum specified number of blocks, because each thread in
 * that kernel can process a variable number of elements.
 */
void get_num_blocks_and_threads(int n, int maxBlocks, int maxThreads, int *blocks, int *threads);

#endif /* NL_BLOCKS_H_ */
