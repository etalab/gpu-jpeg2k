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
