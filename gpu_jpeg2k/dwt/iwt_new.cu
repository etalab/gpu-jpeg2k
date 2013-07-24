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
 * @file iwt.cu
 *
 * @author Milosz Ciznicki
 */
extern "C" {
	#include "iwt_new.h"
}

/**
 * @defgroup I97Coeff Inverse 97 Coefficients.
 *
 * Inverse 97 Coefficients.
 *
 * @{
 */
const float a1 = 1.586134342f;
const float a2 = 0.05298011854f;
const float a3 = -0.8829110762f;
const float a4 = -0.4435068522f;
/** @} */

/**
 * @defgroup IScaleCoeff Inverse scale coefficients.
 *
 * Inverse scale coefficients.
 *
 * @{
 */
const float k = 1.230174104914f;	// 1.230174104914
/** @} */

/**
 * @defgroup I53Coeff Inverse 53 coefficients.
 *
 * Inverse 53 coefficients.
 *
 * @{
 */
//const float p53 = 0.5f;
//const float u53 = -0.25f;
/** @} */

// Template device functions repeated in fwt.cu, because NVCC doesnt support __devices__ functions on different files than kernels.
/**
 * @brief Does inverse lifting process.
 *
 * @param a Coefficient.
 * @param pix_neighborhood Array storing neighbor pixels.
 */
template <class T, unsigned int start, unsigned int end>
__device__ void process_new(const float a, T *pix_neighborhood, int round)
{
#pragma unroll
	for(int i = start; i <= end; i+=2)
	{
		pix_neighborhood[i] += a * (pix_neighborhood[i-1] + pix_neighborhood[i+1] + round);
	}
}

template <class T, unsigned int start, unsigned int end>
__device__ void process2_new(const int sign, const int a, T *pix_neighborhood, int round)
{
#pragma unroll
	for(int i = start; i <= end; i+=2)
	{
		pix_neighborhood[i] += sign * ((pix_neighborhood[i-1] + pix_neighborhood[i+1] + round) >> a);
	}
}

/**
 * @brief Saves results to temporary array.
 *
 * @param p_offset_y Row number actually being processed
 * @param results Array containing temporary results.
 * @param pix_neighborhood Array storing neighbor pixels.
 */
template <class T, unsigned int n, int even, int odd>
__device__ void save_part_results_new(int p_offset_y, T *results, T *pix_neighborhood)
{
#pragma unroll
	for(int i = 0; i < n; i++)
	{
		if(p_offset_y == i)
		{
			results[2*i] = pix_neighborhood[even]; // even - low-pass smaple - a1->a2->a3->a4
			results[2*i + 1] = pix_neighborhood[odd]; // odd - high-pass smaple - a1->a2->a3
		}
	}
}

/**
 * @brief Saves computed results to shared memory.
 *
 * @param k Scale coefficient.
 * @param even_tid Even thread id.
 * @param odd_tid Odd thread id.
 * @param if_tid Thread id.
 * @param p_offset_y Row offset in shared memory.
 * @param p_size_x Computed block width.
 * @param results Array containing computed results.
 * @param shared Shared memory.
 */
template <class T, int sm_size, unsigned int n>
__device__ void save_to_shared_new(float k, short2 even_tid, short2 odd_tid, short if_tid, int p_offset_y, int p_size_x, T *results, T shared[][sm_size])
{
#pragma unroll
	for(int i = 0; i < n; i++)
	{
		if(p_offset_y == i)
		{
			shared[even_tid.x][even_tid.y] = k * results[2*i];
			if(if_tid < p_size_x)// p_size_y
				shared[odd_tid.x][odd_tid.y] = k * results[2*i+1];
		}
	}
}

/**
 * @brief Computes inverse 97 lifting process and saves results to shared memory.
 *
 * @param tidx2 Even thread x id.
 * @param tidy Thread y id.
 * @param p_offset_y Row number actually being processed.
 * @param p_size_x Block summary width.
 * @param p_size_y Block summary height.
 * @param pix_neighborhood Array storing neighbor pixels.
 * @param shared Shared memory.
 * @param results Array containing computed results.
 */
template<int sm_size>
__device__
void iprocess_97_new(const short tidx2, short tidy, short p_offset_y, const short p_size_x, const short p_size_y,
		float *pix_neighborhood, const float shared[][sm_size], float *results)
{
	// Process rows
	while (tidy < p_size_y && tidx2 < p_size_x)
	{
		// Read necessary data
#pragma unroll
		for (int i = 0; i < 9; i++)
		{// pragma unroll
			pix_neighborhood[i] = shared[tidy][tidx2 + i - 4 + OFFSET_97];
		}

		// Update 2
		process_new<float, 2, 8> (a4, pix_neighborhood, 0);

		// Predict 2
		process_new<float, 3, 7> (a3, pix_neighborhood, 0);

		// Update 1
		process_new<float, 4, 6> (a2, pix_neighborhood, 0);

		// Predict 1
		process_new<float, 5, 5> (a1, pix_neighborhood, 0);

		// Can not dynamically index registers, avoid local memory usage.
		//		results[0 + p_offset_y * 2] = pix_neighborhood[4];
		//		results[1 + p_offset_y * 2] = pix_neighborhood[5];
		save_part_results_new<float, (MEMSIZE + (BLOCKSIZEY - 1)) / BLOCKSIZEY, 4, 5> (p_offset_y, results, pix_neighborhood);

		p_offset_y++;
		tidy += BLOCKSIZEY;
	}
}

/**
 * @brief Computes inverse 53 lifting process and saves results to shared memory.
 *
 * @param tidx2 Even thread x id.
 * @param tidy Thread y id.
 * @param p_offset_y Row number actually being processed.
 * @param p_size_x Block summary width.
 * @param p_size_y Block summary height.
 * @param pix_neighborhood Array storing neighbor pixels.
 * @param shared Shared memory.
 * @param results Array containing computed results.
 */
template<int sm_size>
__device__
void iprocess_53_new(const short tidx2, short tidy, short p_offset_y, const short p_size_x, const short p_size_y,
		int *pix_neighborhood, const int shared[][sm_size], int *results)
{
	// Process rows
	while (tidy < p_size_y && tidx2 < p_size_x)
	{
		// Read necessary data
#pragma unroll
		for (int i = 0; i < 6; i++)
		{
			pix_neighborhood[i] = shared[tidy][tidx2 + i - 2 + OFFSET_53];
		}

/*		// Update 1
		process<int, 2, 4> (u53, pix_neighborhood, -2);

		// Predict 1
		process<int, 3, 3> (p53, pix_neighborhood, 0);*/

		// Update 1
		process2_new<int, 2, 4> (-1, 2, pix_neighborhood, 2);

		// Predict 1
		process2_new<int, 3, 3> (1, 1, pix_neighborhood, 0);

		// Can not dynamically index registers, avoid local memory usage.
		//		results[0 + p_offset_y * 2] = pix_neighborhood[4];
		//		results[1 + p_offset_y * 2] = pix_neighborhood[5];
		save_part_results_new<int, (MEMSIZE + (BLOCKSIZEY - 1)) / BLOCKSIZEY, 2, 3> (p_offset_y, results, pix_neighborhood);

		p_offset_y++;
		tidy += BLOCKSIZEY;
	}
}

//#define TEST 1

/**
 * @brief Reads data form global memory to shared memory.
 *
 * Reads data from LL, HL, LH, HH subbands with additional margin containing nesting pixels. The margin width depends on offset variable.
 *
 * @param k Scale coefficient.
 * @param tid Thread id.
 * @param bidx X position of LL and HL block.
 * @param bidy Y position of LL and LH block.
 * @param p_size_x Width of LL and HL block.
 * @param p_size_y Height of LL and LH block.
 * @param ll_sub Size of LL subband.
 * @param img_size Input image size.
 * @param step_x Output image size.
 * @param idata Input array.
 * @param shared Shared memory.
 * @param results Temporary array.
 * @param offset Margin width.
 */
template<class T, int sm_size>
__device__
void read_data_new(const float k, short2 tid, const int2 bidx, const int2 bidy, const short2 p_size_x, const short2 p_size_y,
		const int2 ll_sub, const int2 img_size, const int step_x, const float *idata, T shared[][sm_size], const int offset)
{
	// Threads offset to read margins
	// p_offset_x.x - left block offset
	// p_offset_x.x - right block offset
	short2 p_offset_x;

	// p_offset_y.x - top block offset
	// p_offset_y.x - bottom block offset
	short2 p_offset_y;

	// right offset for LL, LH part
	short right_off_ll_lh;
	// bottom offset for LL, HL part
	short bottom_off_ll_hl;

	// Left and top offset
	// If first block in row, compute left offset to symmetric extension. 2 1 | 0 1 2
	const short p_l_offset_x = ((bidx.x == FIRST_BLOCK) ? (offset - tid.x) : -offset + tid.x);
	// If first block in column, compute top offset to symmetric extension.
	const short p_t_offset_y = ((bidy.x == FIRST_BLOCK) ? (offset - tid.y) : -offset + tid.y);

	// Read patch from GM to SM
	// Do while tid.y and tid.x are smaller than LL part with margins
	while (tid.y < p_size_y.x + 2 * offset)
	{
		while (tid.x < p_size_x.x + 2 * offset)
		{
			// First threads do symmetric extension.
			p_offset_x.x = ((tid.x < offset) ? p_l_offset_x : -offset + tid.x);
			p_offset_y.x = ((tid.y < offset) ? p_t_offset_y : -offset + tid.y);

#ifndef TEST
			// If next to last block in row, compute right offset to symmetric extension. LL, LH part
			right_off_ll_lh
					= ((ll_sub.x - bidx.x < BLOCKSIZEX + offset) ? ((ll_sub.x - bidx.x/* + (ll_sub.x - bidx.x - p_size_x.x) - 2*/-1) - (tid.x
							- (p_size_x.x + offset))) /* Take missing pixel by doing symmetric extensions */: tid.x - offset /* Take pixels from next block */);
			// If next to last block in column, compute bottom offset to symmetric extension. LL, HL part
			bottom_off_ll_hl
					= ((ll_sub.y - bidy.x < BLOCKSIZEY + offset) ? ((ll_sub.y - bidy.x/* + (ll_sub.y - bidy.x - p_size_y.x) - 2*/-1) - (tid.y
							- (p_size_y.x + offset))) /* Take missing pixel by doing  symmetric extensions */: tid.y - offset /* Take pixels from next block */);

			// If next to last block in row, compute right offset to symmetric extension. HL, HH part
			p_offset_x.y
					= ((img_size.x - bidx.y < BLOCKSIZEX + offset) ? ((img_size.x - bidx.y/* + (img_size.x - bidx.y - p_size_x.y) - 2*/-1) - (tid.x
							- (p_size_x.y + offset))) /* Take missing pixel by doing symmetric extensions */: tid.x - offset /* Take pixels from next block */);
			// If next to last block in column, compute bottom offset to symmetric extension. LH, HH part
			p_offset_y.y
					= ((img_size.y - bidy.y < BLOCKSIZEY + offset) ? ((img_size.y - bidy.y/* + (img_size.y - bidy.y - p_size_y.y) - 2*/-1) - (tid.y
							- (p_size_y.y + offset))) /* Take missing pixel by doing  symmetric extensions */: tid.y - offset /* Take pixels from next block */);

#else

			// Take as many pixels as it is possible from right side
			right_off_ll_lh = ((tid.x - offset < ll_sub.x - bidx.x) ? (tid.x - offset) /* Take pixels from next block */: ((ll_sub.x - bidx.x + (ll_sub.x - bidx.x - p_size_x.x) - 2) - (tid.x
					- (p_size_x.x + offset)))) /* Take missing pixel by doing symmetric extensions */;
			// If there are less than offset pixels on bottom side
			bottom_off_ll_hl = ((tid.y - offset < ll_sub.y - bidy.x) ? (tid.y - offset) /* Take pixels from next block */: ((ll_sub.y - bidy.x + (ll_sub.y - bidy.x - p_size_y.x) - 2) - (tid.y
					- (p_size_y.x + offset)))) /* Take missing pixel by doing  symmetric extensions */;

			// If next to last block in row, compute right offset to symmetric extension. LL, LH part
			right_off_ll_lh
					= ((ll_sub.x - bidx.x < BLOCKSIZEX + offset + 1) ? right_off_ll_lh : tid.x - offset /* Take pixels from next block */);
			// If next to last block in column, compute bottom offset to symmetric extension. LL, HL part
			bottom_off_ll_hl
					= ((ll_sub.y - bidy.x < BLOCKSIZEY + offset + 1) ? bottom_off_ll_hl : tid.y - offset /* Take pixels from next block */);

			// If last block in row, compute right offset to symmetric extension.
			right_off_ll_lh = ((ll_sub.x - bidx.x < BLOCKSIZEX + 1) ? ((ll_sub.x - bidx.x - 2) - (tid.x - (p_size_x.x + offset))) /* Symmetric extension 0 1 2 3 | 2 1 0 */
					: right_off_ll_lh);
			// If last block in column, compute bottom offset to symmetric extension.
			bottom_off_ll_hl = ((ll_sub.y - bidy.x < BLOCKSIZEY + 1) ? ((ll_sub.y - bidy.x - 2) - (tid.y - (p_size_y.x + offset))) /* Symmetric extension 0 1 2 3 | 2 1 0 */
					: bottom_off_ll_hl);


//			// If next to last block in row, compute right offset to symmetric extension. LL, LH part
//			right_off_ll_lh
//					= ((ll_sub.x - bidx.x < BLOCKSIZEX + offset + 1) ? ((ll_sub.x - bidx.x + (ll_sub.x - bidx.x - p_size_x.x) - 2) - (tid.x
//							- (p_size_x.x + offset))) /* Take missing pixel by doing symmetric extensions */: tid.x - offset /* Take pixels from next block */);
//			// If next to last block in column, compute bottom offset to symmetric extension. LL, HL part
//			bottom_off_ll_hl
//					= ((ll_sub.y - bidy.x < BLOCKSIZEY + offset + 1) ? ((ll_sub.y - bidy.x + (ll_sub.y - bidy.x - p_size_y.x) - 2) - (tid.y
//							- (p_size_y.x + offset))) /* Take missing pixel by doing  symmetric extensions */: tid.y - offset /* Take pixels from next block */);

			// Take as many pixels as it is possible from right side
			p_offset_x.y = ((tid.x - offset < img_size.x - bidx.y) ? (tid.x - offset) /* Take pixels from next block */: ((img_size.x - bidx.y + (img_size.x - bidx.y - p_size_x.y) - 2) - (tid.x
					- (p_size_x.y + offset)))) /* Take missing pixel by doing symmetric extensions */;
			// If there are less than offset pixels on bottom side
			p_offset_y.y = ((tid.y - offset < img_size.y - bidy.y) ? (tid.y - offset) /* Take pixels from next block */: ((img_size.y - bidy.y + (img_size.y - bidy.y - p_size_y.y) - 2) - (tid.y
					- (p_size_y.y + offset)))) /* Take missing pixel by doing  symmetric extensions */;

			// If next to last block in row, compute right offset to symmetric extension. HL, HH part
			p_offset_x.y
					= ((img_size.x - bidx.y < BLOCKSIZEX + offset + 1) ? p_offset_x.y /* Take missing pixel by doing symmetric extensions */: tid.x - offset /* Take pixels from next block */);
			// If next to last block in column, compute bottom offset to symmetric extension. LH, HH part
			p_offset_y.y
					= ((img_size.y - bidy.y < BLOCKSIZEY + offset + 1) ? p_offset_y.y /* Take missing pixel by doing  symmetric extensions */: tid.y - offset /* Take pixels from next block */);

			// If last block in row, compute right offset to symmetric extension.
			p_offset_x.y = ((img_size.x - bidx.y < BLOCKSIZEX + 1) ? ((img_size.x - bidx.y - 2) - (tid.x - (p_size_x.y + offset))) /* Symmetric extension 0 1 2 3 | 2 1 0 */
					: p_offset_x.y);
			// If last block in column, compute bottom offset to symmetric extension.
			p_offset_y.y = ((img_size.y - bidy.y < BLOCKSIZEY + 1) ? ((img_size.y - bidy.y - 2) - (tid.y - (p_size_y.y + offset))) /* Symmetric extension 0 1 2 3 | 2 1 0 */
					: p_offset_y.y);

#endif

//			// If next to last block in row, compute right offset to symmetric extension. HL, HH part
//			p_offset_x.y
//					= ((img_size.x - bidx.y < BLOCKSIZEX + offset + 1) ? ((img_size.x - bidx.y + (img_size.x - bidx.y - p_size_x.y) - 2) - (tid.x
//							- (p_size_x.y + offset))) /* Take missing pixel by doing symmetric extensions */: tid.x - offset /* Take pixels from next block */);
//			// If next to last block in column, compute bottom offset to symmetric extension. LH, HH part
//			p_offset_y.y
//					= ((img_size.y - bidy.y < BLOCKSIZEY + offset + 1) ? ((img_size.y - bidy.y + (img_size.y - bidy.y - p_size_y.y) - 2) - (tid.y
//							- (p_size_y.y + offset))) /* Take missing pixel by doing  symmetric extensions */: tid.y - offset /* Take pixels from next block */);

//			// If last right part of the block in row is smaller than left part of the block, shift offset
//			p_offset_x.y = ((p_size_x.y < p_size_x.x) ? ((img_size.x - bidx.y - 3) - (tid.x - (p_size_x.x + offset))) /* Symmetric extension 0 1 2 3 | 2 1 0 */
//			: p_offset_x.y);
//			// If last bottom part of the block in column is smaller than top part of the block, shift offset
//			p_offset_y.y = ((p_size_y.y < p_size_y.x) ? ((img_size.y - bidy.y - 3) - (tid.y - (p_size_y.x + offset))) /* Symmetric extension 0 1 2 3 | 2 1 0 */
//			: p_offset_y.y);

			// Last threads do symmetric extension.
			p_offset_x.x = ((tid.x >= p_size_x.x + offset) ? right_off_ll_lh : p_offset_x.x);
			p_offset_y.x = ((tid.y >= p_size_y.x + offset) ? bottom_off_ll_hl : p_offset_y.x);

			// Last threads do symmetric extension.
			p_offset_x.y = ((tid.x >= p_size_x.y + offset) ? p_offset_x.y : p_offset_x.x);
			p_offset_y.y = ((tid.y >= p_size_y.y + offset) ? p_offset_y.y : p_offset_y.x);

			/* Rotate and mix values */
			/* LL - top left part block */
			shared[2 * tid.x][2 * tid.y] = k * idata[bidx.x + p_offset_x.x + (bidy.x + p_offset_y.x) * step_x];
			/* HL - top right part block */
			if (bidx.y + p_offset_x.y < img_size.x/* && bidy.x + p_offset_y.x < img_size.y*/)
				shared[2 * tid.x + 1/* + p_size_y.x + OFFSET_97*/][2 * tid.y] = k * idata[bidx.y + p_offset_x.y + (bidy.x
						+ p_offset_y.x) * step_x];
			/* LH - bottom left part block */
			if (/*bidx.x + p_offset_x.x < img_size.x && */bidy.y + p_offset_y.y < img_size.y)
				shared[2 * tid.x][2 * tid.y + 1/* + p_size_x.x + OFFSET_97*/] = idata[bidx.x + p_offset_x.x + (bidy.y
						+ p_offset_y.y) * step_x] / k;
			/* HH - bottom right part block */
			if (bidx.y + p_offset_x.y < img_size.x && bidy.y + p_offset_y.y < img_size.y)
				shared[2 * tid.x + 1/* + p_size_y.x + OFFSET_97*/][2 * tid.y + 1/* + p_size_x.x + OFFSET_97*/] = idata[bidx.y + p_offset_x.y + (bidy.y + p_offset_y.y) * step_x] / k;

			tid.x += BLOCKSIZEX;
		}
		tid.x = threadIdx.x;
		tid.y += BLOCKSIZEY;
	}
}

/**
 * @brief Saves data from shared memory to global memory.
 *
 * @param tid Thread id.
 * @param p_size_sum Block summary width and height.
 * @param bidx_x Location of left block.
 * @param bidy_x Location of right block.
 * @param img_size Input image size.
 * @param step_x Ouput image size.
 * @param odata Output array.
 * @param shared Shared memory.
 */
template<class T, int sm_size>
__device__
void save_data_new(short2 tid, const short2 p_size_sum, const int bidx_x, const int bidy_x, const int2 img_size, const int step_x,
		float *odata, const T shared[][sm_size])
{
	// Save to GM
	while (tid.y < p_size_sum.y)
	{
		while (tid.x < p_size_sum.x)
		{
			if (2 * bidx_x + tid.x < img_size.x && 2 * bidy_x + tid.y < img_size.y)
				odata[2 * bidx_x + tid.x + (2 * bidy_x + tid.y) * step_x] = shared[tid.y][tid.x];

			tid.x += BLOCKSIZEX;
		}
		tid.x = threadIdx.x;
		tid.y += BLOCKSIZEY;
	}
}

/**
 * @brief Computes inverse wavelet transform 97.
 *
 * @param idata Input data.
 * @param odata Output data
 * @param img_size Struct with input image width and height.
 * @param step Struct with output image width and height.
 */
__global__
void iwt97_new(const float *idata, float *odata, const int2 img_size, const int2 step)
{
	// Shared memory for part of the signal
	__shared__ float shared[MEMSIZE][MEMSIZE + 1];

	// LL subband dimensions
//	const int2 ll_sub = make_int2((int) ceilf(img_size.x / 2.0), (int) ceilf(img_size.y / 2.0));
	const int2 ll_sub = make_int2((img_size.x + 1) >> 1, (img_size.y + 1) >> 1);

	// Input x, y block dimension
	// bidx.x - left block location
	// bidx.y - right block location
	const int2 bidx = make_int2(blockIdx.x * BLOCKSIZEX, ll_sub.x + blockIdx.x * BLOCKSIZEX);
	// bidy.x - top block location
	// bidy.y - bottom block location
	const int2 bidy = make_int2(blockIdx.y * BLOCKSIZEY, ll_sub.y + blockIdx.y * BLOCKSIZEY);

	// Even thread id
	const short tidx2 = threadIdx.x * 2;

	// thread id
	short2 tid = make_short2(threadIdx.x, threadIdx.y);

	// Patch size
	/* Compute patch offset and size */
	// p_size_x.x - left part block x size
	// p_size_x.y - right part block x size
	const short2 p_size_x = make_short2(ll_sub.x - bidx.x < BLOCKSIZEX ? ll_sub.x - bidx.x : BLOCKSIZEX,
			img_size.x - bidx.y < BLOCKSIZEX ? img_size.x - bidx.y : BLOCKSIZEX);

	// p_size_y.x - top part block x size
	// p_size_y.y - bottom part block x size
	const short2 p_size_y = make_short2(ll_sub.y - bidy.x < BLOCKSIZEY ? ll_sub.y - bidy.x : BLOCKSIZEY,
			img_size.y - bidy.y < BLOCKSIZEY ? img_size.y - bidy.y : BLOCKSIZEY);

	// summary size
	const short2 p_size_sum = make_short2(p_size_x.x + p_size_x.y, p_size_y.x + p_size_y.y); /* block x size */

	// Threads offset to read margins
	short p_offset_y_t;
	// Allocate registers in order to compute even and odd pixels.
	float pix_neighborhood[9];
	// Minimize registers usage. Right | bottom offset. Odd | even result pixels.
	float results[6];

	read_data_new<float, MEMSIZE + 1>(k, tid, bidx, bidy, p_size_x, p_size_y, ll_sub, img_size, step.x, idata, shared, OFFSET_97/2);

	__syncthreads();

	// thread x id
	tid.x = threadIdx.x;
	// thread y id
	tid.y = threadIdx.y;

	// Row number
	p_offset_y_t = 0;

	// Process columns
	iprocess_97_new<MEMSIZE + 1>(tidx2, tid.y, p_offset_y_t, p_size_sum.y, p_size_sum.x + 4 * IOFFSET_97, pix_neighborhood, shared, results);

	__syncthreads();

	tid.x = threadIdx.x;
	tid.y = threadIdx.y;
	p_offset_y_t = 0;

	// safe results and rotate
	while (tid.y < p_size_sum.x + 4 * IOFFSET_97 && 2 * tid.x < p_size_sum.y)
	{
		// Can not dynamically index registers, avoid local memory usage.
		//		shared[tid.x][tid.y] = k2 * results[0 + p_offset_y * 2];
		//		if(tid.x + BLOCKSIZEX < p_size_sum.y)
		//			shared[tid.x + BLOCKSIZEX][tid.y] = k1 * results[1 + p_offset_y * 2];
		if (tid.y % 2 == 0)
		{
			save_to_shared_new<float, MEMSIZE + 1, (MEMSIZE + (BLOCKSIZEY - 1)) / BLOCKSIZEY> (k, make_short2(2 * tid.x, tid.y),
					make_short2(2 * tid.x + 1, tid.y), 2 * tid.x + 1, p_offset_y_t, p_size_sum.y, results, shared);

		} else
		{
			save_to_shared_new<float, MEMSIZE + 1, (MEMSIZE + (BLOCKSIZEY - 1)) / BLOCKSIZEY> (1/k, make_short2(2 * tid.x, tid.y),
					make_short2(2 * tid.x + 1, tid.y), 2 * tid.x + 1, p_offset_y_t, p_size_sum.y, results, shared);
		}
		p_offset_y_t++;
		tid.y += BLOCKSIZEY;
	}
	__syncthreads();

	tid.x = threadIdx.x;
	tid.y = threadIdx.y;

	// Row number
	p_offset_y_t = 0;

	// Process rows
	iprocess_97_new<MEMSIZE + 1>(tidx2, tid.y, p_offset_y_t, p_size_sum.x, p_size_sum.y, pix_neighborhood, shared, results);

	__syncthreads();

	tid.x = threadIdx.x;
	tid.y = threadIdx.y;

	// Row number
	p_offset_y_t = 0;

	// Safe results
	while (2 * tid.x < p_size_sum.x && tid.y < p_size_sum.y)
	{
		// Can not dynamically index registers, avoid local memory usage.
		//		shared[tid.x][tid.y] = k2 * results[0 + p_offset_y * 2];
		//		if(tid.x + BLOCKSIZEX < p_size_sum.y)
		//			shared[tid.x + BLOCKSIZEX][tid.y] = k1 * results[1 + p_offset_y * 2];
		save_to_shared_new<float, MEMSIZE + 1, (MEMSIZE + (BLOCKSIZEY - 1)) / BLOCKSIZEY> (1, make_short2(tid.y, 2 * tid.x), make_short2(tid.y, 2 * tid.x + 1), 2
				* tid.x + 1, p_offset_y_t, p_size_sum.x, results, shared);

		p_offset_y_t++;
		tid.y += BLOCKSIZEY;
	}
	__syncthreads();

	tid.x = threadIdx.x;
	tid.y = threadIdx.y;

	save_data_new<float, MEMSIZE + 1>(tid, p_size_sum, bidx.x, bidy.x, img_size, step.x, odata, shared);
}

/**
 * @brief Computes inverse wavelet transform 53.
 *
 * @param idata Input data.
 * @param odata Output data
 * @param img_size Struct with input image width and height.
 * @param step Struct with output image width and height.
 */
__global__
void iwt53_new(const float *idata, float *odata, const int2 img_size, const int2 step)
{
	// shared memory for part of the signal
	__shared__ int shared[MEMSIZE][MEMSIZE + 1];

	// LL subband dimensions - ceil of input image dimensions
//	const int2 ll_sub = make_int2((int) ceilf(img_size.x / 2.0), (int) ceilf(img_size.y / 2.0));
	const int2 ll_sub = make_int2((img_size.x + 1) >> 1, (img_size.y + 1) >> 1);

	// Input x, y block dimension
	// Width
	// bidx.x - left block
	// bidx.y - right block
	const int2 bidx = make_int2(blockIdx.x * BLOCKSIZEX, ll_sub.x + blockIdx.x * BLOCKSIZEX);
	// Height
	// bidy.x - top block
	// bidy.y - bottom block
	const int2 bidy = make_int2(blockIdx.y * BLOCKSIZEY, ll_sub.y + blockIdx.y * BLOCKSIZEY);

	// Even thread id
	const short tidx2 = threadIdx.x * 2;

	// thread id
	short2 tid = make_short2(threadIdx.x, threadIdx.y);

	// Patch size
	/* Compute patch offset and size */
	// p_size_x.x - left part block x size
	// p_size_x.y - right part block x size
	const short2 p_size_x = make_short2(ll_sub.x - bidx.x < BLOCKSIZEX ? ll_sub.x - bidx.x : BLOCKSIZEX,
			img_size.x - bidx.y < BLOCKSIZEX ? img_size.x - bidx.y : BLOCKSIZEX);

	// p_size_y.x - top part block x size
	// p_size_y.y - bottom part block x size
	const short2 p_size_y = make_short2(ll_sub.y - bidy.x < BLOCKSIZEY ? ll_sub.y - bidy.x : BLOCKSIZEY,
			img_size.y - bidy.y < BLOCKSIZEY ? img_size.y - bidy.y : BLOCKSIZEY);

	// summary size
	const short2 p_size_sum = make_short2(p_size_x.x + p_size_x.y, p_size_y.x + p_size_y.y); /* block x size */

	// Threads offset to read margins
	short p_offset_y_t;
	// Allocate registers in order to compute even and odd pixels.
	int pix_neighborhood[6];
	// Minimize registers usage. Right | bottom offset. Odd | even result pixels.
	int results[6];

	read_data_new<int, MEMSIZE + 1>(1, tid, bidx, bidy, p_size_x, p_size_y, ll_sub, img_size, step.x, idata, shared, OFFSET_53/2);

	__syncthreads();

	// thread x id
	tid.x = threadIdx.x;
	// thread y id
	tid.y = threadIdx.y;

	// Row number
	p_offset_y_t = 0;

	// Process columns
	iprocess_53_new<MEMSIZE + 1>(tidx2, tid.y, p_offset_y_t, p_size_sum.y, p_size_sum.x + 2 * OFFSET_53, pix_neighborhood, shared, results);

	__syncthreads();

	tid.x = threadIdx.x;
	tid.y = threadIdx.y;
	p_offset_y_t = 0;

	// safe results and rotate
	while (tid.y < p_size_sum.x + 2 * OFFSET_53 && 2 * tid.x < p_size_sum.y)
	{
		// Can not dynamically index registers, avoid local memory usage.
		//		shared[tid.x][tid.y] = k2 * results[0 + p_offset_y * 2];
		//		if(tid.x + BLOCKSIZEX < p_size_sum.y)
		//			shared[tid.x + BLOCKSIZEX][tid.y] = k1 * results[1 + p_offset_y * 2];
		save_to_shared_new<int, MEMSIZE + 1, (MEMSIZE + (BLOCKSIZEY - 1)) / BLOCKSIZEY> (1, make_short2(2 * tid.x, tid.y), make_short2(2 * tid.x + 1, tid.y), 2
				* tid.x + 1, p_offset_y_t, p_size_sum.y, results, shared);

		p_offset_y_t++;
		tid.y += BLOCKSIZEY;
	}
	__syncthreads();

	tid.x = threadIdx.x;
	tid.y = threadIdx.y;

	// Row number
	p_offset_y_t = 0;

	// Process rows
	iprocess_53_new<MEMSIZE + 1>(tidx2, tid.y, p_offset_y_t, p_size_sum.x, p_size_sum.y, pix_neighborhood, shared, results);

	__syncthreads();

	tid.x = threadIdx.x;
	tid.y = threadIdx.y;

	// Row number
	p_offset_y_t = 0;

	// Safe results
	while (2 * tid.x < p_size_sum.x && tid.y < p_size_sum.y)
	{
		// Can not dynamically index registers, avoid local memory usage.
		//		shared[tid.x][tid.y] = k2 * results[0 + p_offset_y * 2];
		//		if(tid.x + BLOCKSIZEX < p_size_sum.y)
		//			shared[tid.x + BLOCKSIZEX][tid.y] = k1 * results[1 + p_offset_y * 2];
		save_to_shared_new<int, MEMSIZE + 1, (MEMSIZE + (BLOCKSIZEY - 1)) / BLOCKSIZEY> (1, make_short2(tid.y, 2 * tid.x), make_short2(tid.y, 2 * tid.x + 1), 2
				* tid.x + 1, p_offset_y_t, p_size_sum.x, results, shared);

		p_offset_y_t++;
		tid.y += BLOCKSIZEY;
	}
	__syncthreads();

	tid.x = threadIdx.x;
	tid.y = threadIdx.y;

	// Save to GM
	save_data_new<int, MEMSIZE + 1>(tid, p_size_sum, bidx.x, bidy.x, img_size, step.x, odata, shared);
}

