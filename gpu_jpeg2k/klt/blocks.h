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
