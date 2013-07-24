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
 * nl_blocks.c
 *
 *  Created on: 2010-05-16
 *      Author: Milosz Ciznicki
 */

#include "blocks.h"

int min(int a, int b) {
	return a < b ? a : b;
}

/*
 * @brief Get next power of two.
 */
unsigned int next_pow2( unsigned int x ) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

/*
 * @brief Compute the number of threads and blocks to use for the given reduction kernel
 * Set threads / block to the minimum of maxThreads and
 * n/2. Observe the maximum specified number of blocks, because each thread in
 * that kernel can process a variable number of elements.
 */
void get_num_blocks_and_threads(int n, int maxBlocks, int maxThreads, int *blocks, int *threads)
{
	*threads = (n < maxThreads*2) ? next_pow2((n + 1)/ 2) : maxThreads;
	*blocks = (n + (*threads * 2 - 1)) / (*threads * 2);
	*blocks = min(maxBlocks, *blocks);
}
