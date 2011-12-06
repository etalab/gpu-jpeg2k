/**
 * @file fwt_new.cu
 *
 * @author Milosz Ciznicki
 */

/**
 * @file fwt.cu
 *
 * @author Milosz Ciznicki
 */

extern "C" {
	#include "fwt_new.h"
}

/**
 * @defgroup 97Coeff 97 Coefficients.
 *
 * 97 Coefficients.
 *
 * @{
 */
const float a1 = -1.586134342f;
const float a2 = -0.05298011854f;
const float a3 = 0.8829110762f;
const float a4 = 0.4435068522f;
/** @} */

/**
 * @defgroup ScaleCoeff Scale coefficients.
 *
 * Scale coefficients.
 *
 * @{
 */
const float k = 1.230174104914f;	// 1.230174104914
/** @} */

/**
 * @defgroup 53Coeff 53 coefficients.
 *
 * 53 coefficients.
 *
 * @{
 */
const float p53 = -0.5f;
const float u53 = 0.25f;
/** @} */

/**
 * @brief Does lifting process.
 *
 * @param a Coefficient.
 * @param pix_neighborhood Array storing neighbor pixels.
 */
template <class T, unsigned int start, unsigned int end>
__device__
void process_new(const float a, T *pix_neighborhood)
{
#pragma unroll
	for(int i = start; i <= end; i+=2)
	{
		pix_neighborhood[i] += a * (pix_neighborhood[i-1] + pix_neighborhood[i+1]);
	}
}

/**
 * @brief Does lifting process.
 *
 * @param a Coefficient.
 * @param pix_neighborhood Array storing neighbor pixels.
 */
template <class T, unsigned int start, unsigned int end>
__device__
void process53_new(const int sign, const int approx, const int a, T *pix_neighborhood)
{
#pragma unroll
	for(int i = start; i <= end; i+=2)
	{
		pix_neighborhood[i] += sign * ((pix_neighborhood[i-1] + pix_neighborhood[i+1] + approx) >> a);
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
__device__
void save_part_results_new(int p_offset_y, T *results, T *pix_neighborhood)
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

template <class T, unsigned int n>
__device__
void save_to_shared2_new(float k, short2 tid, short2 p_offset, int p_size_x, T *results, T shared[][MEMSIZE + 1])
{
#pragma unroll
	for(int i = 0; i < n; i++)
	{
		if(p_offset.y == i)
		{
			shared[tid.y][tid.x] = results[2*i] / k;
			if(tid.x + p_offset.x < p_size_x)// p_size_y
				shared[tid.y][tid.x + p_offset.x] = k * results[2*i + 1];
		}
	}
}

/**
 * @brief Saves computed results to shared memory.
 *
 * @param k Scale coefficient.
 * @param tid Thread id.
 * @param p_offset Offset in shared memory.
 * @param p_size_x Computed block width.
 * @param results Array containing computed results.
 * @param shared Shared memory.
 */
template <class T, unsigned int n>
__device__
void save_to_shared_new(float k, short2 tid, short2 p_offset, int p_size_x, T *results, T shared[][MEMSIZE + 1])
{
#pragma unroll
	for(int i = 0; i < n; i++)
	{
		if(p_offset.y == i)
		{
			shared[tid.y][tid.x] = results[2*i] / k;
			if(tid.y + p_offset.x < p_size_x)// p_size_y
				shared[tid.y + p_offset.x][tid.x] = k * results[2*i + 1];
		}
	}
}

/**
 * @brief Computes forward 97 lifting process and saves results to shared memory.
 *
 * @param tidy Thread y id.
 * @param tidx2 Even thread x id.
 * @param p_offset_y Row number actually being processed.
 * @param pix_neighborhood Array storing neighbor pixels.
 * @param shared Shared memory.
 * @param results Array containing computed results.
 */
__device__
void fprocess_97_new(short tidy, const short tidx2, short p_offset_y, float *pix_neighborhood, const float shared[][MEMSIZE + 1],
		float *results)
{
	// Read necessary data
#pragma unroll
	for (int i = 0; i < 9; i++)
	{// pragma unroll
		pix_neighborhood[i] = shared[tidy][tidx2 + i - 4 + OFFSET_97];
	}

	// Predict 1
	process_new<float, 1, 7> (a1, pix_neighborhood);

	// Update 1
	process_new<float, 2, 6> (a2, pix_neighborhood);

	// Predict 2
	process_new<float, 3, 5> (a3, pix_neighborhood);

	// Update 2
	process_new<float, 4, 4> (a4, pix_neighborhood);

	// Can not dynamically index registers, avoid local memory usage.
	//		results[0 + p_offset_y * 2] = pix_neighborhood[4];
	//		results[1 + p_offset_y * 2] = pix_neighborhood[5];
	save_part_results_new<float, (MEMSIZE + (BLOCKSIZEY - 1)) / BLOCKSIZEY, 4, 5> (p_offset_y, results, pix_neighborhood);
}

/**
 * @brief Computes forward 53 lifting process and saves results to shared memory.
 *
 * @param tidy Thread y id.
 * @param tidx2 Even thread x id.
 * @param p_offset_y Row number actually being processed.
 * @param pix_neighborhood Array storing neighbor pixels.
 * @param shared Shared memory.
 * @param results Array containing computed results.
 */
__device__
void fprocess_53_2_new(short tidy, const short tidx2, short p_offset_y, int *pix_neighborhood, const int shared[][MEMSIZE + 1],
		int *results)
{
	// Read necessary data
#pragma unroll
	for (int i = 0; i < 5; i++)
	{
		pix_neighborhood[i] = shared[tidy][tidx2 + i - 2 + OFFSET_53];
	}

	// Predict 1
//	process53<int, 1, 3> (-1, 0, 1, pix_neighborhood);
	pix_neighborhood[1] -= ((pix_neighborhood[0] + pix_neighborhood[2]) >> 1);
	pix_neighborhood[3] -= ((pix_neighborhood[2] + pix_neighborhood[4]) >> 1);
	// Update 1
//	process53<int, 2, 2> (1, 2, 2, pix_neighborhood);
	pix_neighborhood[2] += ((pix_neighborhood[1] + pix_neighborhood[3] + 2) >> 2);

	// Can not dynamically index registers, avoid local memory usage.
	//		results[0 + p_offset_y * 2] = pix_neighborhood[4];
	//		results[1 + p_offset_y * 2] = pix_neighborhood[5];
	save_part_results_new<int, (MEMSIZE + (BLOCKSIZEY - 1)) / BLOCKSIZEY, 2, 3> (p_offset_y, results, pix_neighborhood);
}

/**
 * @brief Computes forward 53 lifting process and saves results to shared memory.
 *
 * @param tidy Thread y id.
 * @param tidx2 Even thread x id.
 * @param p_offset_y Row number actually being processed.
 * @param pix_neighborhood Array storing neighbor pixels.
 * @param shared Shared memory.
 * @param results Array containing computed results.
 */
__device__
void fprocess_53_new(short tidy, const short tidx2, short p_offset_y, int *pix_neighborhood, const int shared[][MEMSIZE + 1],
		int *results)
{
	// Read necessary data
#pragma unroll
	for (int i = 0; i < 5; i++)
	{
		pix_neighborhood[i] = shared[tidy][tidx2 + i - 2 + OFFSET_53];
	}

	// Predict 1
	process_new<int, 1, 3> (p53, pix_neighborhood);
	// Update 1
	process_new<int, 2, 2> (u53, pix_neighborhood);

	// Can not dynamically index registers, avoid local memory usage.
	//		results[0 + p_offset_y * 2] = pix_neighborhood[4];
	//		results[1 + p_offset_y * 2] = pix_neighborhood[5];
	save_part_results_new<int, (MEMSIZE + (BLOCKSIZEY - 1)) / BLOCKSIZEY, 2, 3> (p_offset_y, results, pix_neighborhood);
}

/**
 * @brief Reads data form global memory to shared memory.
 *
 * Data from global memory is read with additional margin containing nesting pixels. The margin width depends on offset variable.
 *
 * @param tid Thread id.
 * @param bid Block id.
 * @param p_size Block width and height.
 * @param img_size Input image width and height.
 * @param step_x Output image width.
 * @param idata Input array.
 * @param shared Shared memory
 * @param results Temporary array.
 * @param offset Margin width.
 */
template<class T>
__device__
void read_data_new(short2 tid, const int2 bid, const short2 p_size, const int2 img_size, const int step_x, const float *idata,
		T shared[][MEMSIZE + 1], T *results, const int offset)
{
	// Threads offset to read margins
	short2 p_offset;
	// Left and top offset
	// If first block in row, compute left offset to symmetric extension.
	const short p_l_offset_x =
			((bid.x == FIRST_BLOCK) ? (offset - tid.x) /* left symmetric extension*/: -offset + tid.x /* take from previous block */);
	// If first block in column, compute top offset to symmetric extension.
	const short p_t_offset_y =
			((bid.y == FIRST_BLOCK) ? (offset - tid.y) /* top symmetric extension*/: -offset + tid.y /* take from previous block */);

	// Read patch from GM to SM
	while (tid.y < p_size.y + 2 * offset)
	{
		while (tid.x < p_size.x + 2 * offset)
		{
			// First offset threads do symmetric extension.
			p_offset.x = ((tid.x < offset) ? p_l_offset_x /* take from left adjacent block */: -offset + tid.x /* take normally pixels */);
			p_offset.y = ((tid.y < offset) ? p_t_offset_y /* take from top adjacent block */: -offset + tid.y /* take normally pixels */);

			// Take as many pixels as it is possible from right side
			results[2] = ((tid.x - offset < img_size.x - bid.x) ? (tid.x - offset) /* Take pixels from next block */: ((img_size.x - bid.x
					+ (img_size.x - bid.x - p_size.x) - 2) - (tid.x - (p_size.x + offset)))) /* Take missing pixel by doing symmetric extensions */;
			// If there are less than offset pixels on bottom side
			results[3] = ((tid.y - offset < img_size.y - bid.y) ? (tid.y - offset) /* Take pixels from next block */: ((img_size.y - bid.y
					+ (img_size.y - bid.y - p_size.y) - 2) - (tid.y - (p_size.y + offset)))) /* Take missing pixel by doing  symmetric extensions */;

//			// If next to last block in row, compute right offset to symmetric extension.
//			results[2] = ((img_size.x - (bid.x + PATCHX) < offset) ? ((img_size.x - bid.x - 2) - (tid.x - (p_size.x + offset))) /* Take missing pixel by doing symmetric extensions */ : tid.x - offset /* Take pixels from next block */);
//			// If next to last block in column, compute bottom offset to symmetric extension.
//			results[3] = ((img_size.y - (bid.y + PATCHY) < offset) ? ((img_size.y - bid.y - 2) - (tid.y - (p_size.y + offset))) /* Take missing pixel by doing  symmetric extensions */ : tid.y - offset /* Take pixels from next block */);

//			// If next to last block in row, compute right offset to symmetric extension.
//			results[2] = ((img_size.x - (bid.x + PATCHX) < offset) ? ((img_size.x - bid.x - 1) - (tid.x - (p_size.x + offset))) : tid.x - offset /* Take pixels from next block */);
//			// If next to last block in column, compute bottom offset to symmetric extension.
//			results[3] = ((img_size.y - (bid.y + PATCHY) < offset) ? ((img_size.y - bid.y - 1) - (tid.y - (p_size.y + offset))) : tid.y - offset /* Take pixels from next block */);

			// If next to last block in row, compute right offset to symmetric extension.
			results[2] = ((img_size.x - (bid.x + PATCHX) < offset) ? results[2] : tid.x - offset /* Take pixels from next block */);
			// If next to last block in column, compute bottom offset to symmetric extension.
			results[3] = ((img_size.y - (bid.y + PATCHY) < offset) ? results[3] : tid.y - offset /* Take pixels from next block */);

//			// If next to last block in row, compute right offset to symmetric extension.
//			results[2] = ((img_size.x - bid.x < PATCHX + 4) ? results[2] : tid.x - offset /* Take pixels from next block */);
//			// If next to last block in column, compute bottom offset to symmetric extension.
//			results[3] = ((img_size.y - bid.y < PATCHY + 4) ? results[3] : tid.y - offset /* Take pixels from next block */);

			// If last block in row, compute right offset to symmetric extension.
			results[2] = ((img_size.x - bid.x < PATCHX + 1) ? ((img_size.x - bid.x - 2) - (tid.x - (p_size.x + offset))) /* Symmetric extension 0 1 2 3 | 2 1 0 */
					: results[2]);
			// If last block in column, compute bottom offset to symmetric extension.
			results[3] = ((img_size.y - bid.y < PATCHY + 1) ? ((img_size.y - bid.y - 2) - (tid.y - (p_size.y + offset))) /* Symmetric extension 0 1 2 3 | 2 1 0 */
					: results[3]);

			// Last threads do symmetric extension.
			p_offset.x = ((tid.x >= p_size.x + offset) ? results[2] : p_offset.x);
			p_offset.y = ((tid.y >= p_size.y + offset) ? results[3] : p_offset.y);

			shared[tid.y][tid.x] = idata[bid.x + p_offset.x + (bid.y + p_offset.y) * step_x];

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
 * @param p_offset X and y offset in shared memory.
 * @param p_size Width and height computed block.
 * @param img_size Image width and height.
 * @param step_x Output image width.
 * @param odata Output array.
 * @param shared Shared memory.
 */
template<class T>
__device__
void save_data_new(short2 tid, short2 p_offset, const short2 p_size, const int2 img_size, const int step_x, float *odata,
		T shared[][MEMSIZE + 1])
{
	tid.x = threadIdx.x;
	tid.y = threadIdx.y;

	// Column offset
	p_offset.x = (int) ceilf(p_size.x / 2.0f);
	// Row offset
	p_offset.y = (int) ceilf(p_size.y / 2.0f);

	// Save to GM
	while (tid.y < p_offset.y)
	{
		//if(tid.x + blockIdx.x * BLOCKSIZEX < img_size.x && tid.y + blockIdx.y * BLOCKSIZEY < img_size.y)
		while (tid.x < p_offset.x)
		{
			odata[tid.x + blockIdx.x * PATCHX_DIV_2 + (tid.y + blockIdx.y * PATCHY_DIV_2) * step_x] = shared[tid.y][tid.x];
			if (tid.x + (int) ceilf(img_size.x / 2.0f) + blockIdx.x * PATCHX_DIV_2 < img_size.x)
				odata[tid.x + (int) ceilf(img_size.x / 2.0f) + blockIdx.x * PATCHX_DIV_2 + (tid.y + blockIdx.y * PATCHY_DIV_2) * step_x]
						= shared[tid.y][tid.x + p_offset.x];
			if (tid.y + (int) ceilf(img_size.y / 2.0f) + blockIdx.y * PATCHY_DIV_2 < img_size.y)
				odata[tid.x + blockIdx.x * PATCHX_DIV_2 + (tid.y + (int) ceilf(img_size.y / 2.0f) + blockIdx.y * PATCHY_DIV_2) * step_x]
						= shared[tid.y + p_offset.y][tid.x];
			if (tid.x + (int) ceilf(img_size.x / 2.0f) + blockIdx.x * PATCHX_DIV_2 < img_size.x && tid.y + (int) ceilf(img_size.y / 2.0f)
					+ blockIdx.y * PATCHY_DIV_2 < img_size.y)
				odata[tid.x + (int) ceilf(img_size.x / 2.0f) + blockIdx.x * PATCHX_DIV_2 + (tid.y + (int) ceilf(img_size.y / 2.0f)
						+ blockIdx.y * PATCHY_DIV_2) * step_x] = shared[tid.y + p_offset.y][tid.x + p_offset.x];

			tid.x += BLOCKSIZEX;
		}
		tid.x = threadIdx.x;
		tid.y += BLOCKSIZEY;
	}
}

/**
 * @brief Computes forward wavelet transform 97.
 *
 * @param idata Input data.
 * @param odata Output data
 * @param img_size Struct with input image width and height.
 * @param step Struct with output image width and height.
 */
__global__
void fwt97_new(const float *idata, float *odata, const int2 img_size, const int2 step)
{
	/* Shared memory for part of the signal */
	__shared__ float shared[MEMSIZE][MEMSIZE + 1];

	/* Input x, y block dimension */
	const int2 bid = make_int2(blockIdx.x * PATCHX, blockIdx.y * PATCHY);

	/* Thread id */
	short2 tid = make_short2(threadIdx.x, threadIdx.y);

	/* Threads offset to read margins */
	short2 p_offset;

	// Patch size
	/* Compute patch offset and size */
	const short2 p_size = make_short2(img_size.x - bid.x < PATCHX ? img_size.x - bid.x : PATCHX, img_size.y - bid.y < PATCHY ? img_size.y
			- bid.y : PATCHY);

	/* Even thread id */
//	const short tidx2 = threadIdx.x * 2;

	/* Allocate registers in order to compute even and odd pixels. */
	float pix_neighborhood[9];
	/* Minimize registers usage. Right | bottom offset. Odd | even result pixels. */
	float results[((MEMSIZE + (BLOCKSIZEY - 1)) / BLOCKSIZEY) * 2];

	read_data_new<float>(tid, bid, p_size, img_size, step.x, idata, shared, results, OFFSET_97);

	__syncthreads();

	// Thread x id
	tid.x = threadIdx.x;
	// Thread y id
	tid.y = threadIdx.y;

	// Row number
	p_offset.y = 0;

	// Process rows
	while (tid.y * 2 < p_size.y && tid.x < p_size.x + 2 * OFFSET_97)
	{
//		fprocess_97(tid.y, tidx2, p_offset.y, pix_neighborhood, shared, results);

		// Read necessary data
		#pragma unroll
		for (int i = 0; i < 9; i++)
		{// pragma unroll
			pix_neighborhood[i] = shared[tid.y * 2 + i - 4 + OFFSET_97][tid.x];
		}

		// Predict 1
		process_new<float, 1, 7> (a1, pix_neighborhood);

		// Update 1
		process_new<float, 2, 6> (a2, pix_neighborhood);

		// Predict 2
		process_new<float, 3, 5> (a3, pix_neighborhood);

		// Update 2
		process_new<float, 4, 4> (a4, pix_neighborhood);

		// Can not dynamically index registers, avoid local memory usage.
		//		results[0 + p_offset_y * 2] = pix_neighborhood[4];
		//		results[1 + p_offset_y * 2] = pix_neighborhood[5];
		save_part_results_new<float, (MEMSIZE + (BLOCKSIZEY - 1)) / BLOCKSIZEY, 4, 5> (p_offset.y, results, pix_neighborhood);

		p_offset.y++;
		tid.x += BLOCKSIZEY;
	}
//	// Process rows
//	while (tid.y < p_size.y + 2 * OFFSET_97 && tidx2 < p_size.x)
//	{
//		fprocess_97(tid.y, tidx2, p_offset.y, pix_neighborhood, shared, results);
//
//		p_offset.y++;
//		tid.y += BLOCKSIZEY;
//	}
	__syncthreads();

	tid.x = threadIdx.x;
	tid.y = threadIdx.y;

	p_offset.y = 0;
	// Column offset
	p_offset.x = (int) ceilf(p_size.y / 2.0f);

	// safe results and rotate
	while (tid.y < p_size.y && tid.x < p_size.x + 2 * OFFSET_97)
	{
		// Can not dynamically index registers, avoid local memory usage.
		//		shared[tid.x][tid.y] = k2 * results[0 + p_offset.y * 2];
		//		if(tid.x + BLOCKSIZEX < p_size.y)
		//			shared[tid.x + BLOCKSIZEX][tid.y] = k1 * results[1 + p_offset.y * 2];
		save_to_shared_new<float, (MEMSIZE + (BLOCKSIZEY - 1)) / BLOCKSIZEY> (k, tid, p_offset, p_size.y, results, shared);

		p_offset.y++;
		tid.x += BLOCKSIZEY;
	}
	__syncthreads();

	tid.x = threadIdx.x;
	tid.y = threadIdx.y;

	// Row number
	p_offset.y = 0;

	// Process columns
	while (tid.y < p_size.y && tid.x * 2 < p_size.x)
	{
//		fprocess_97(tid.y, tidx2, p_offset.y, pix_neighborhood, shared, results);

		// Read necessary data
		#pragma unroll
		for (int i = 0; i < 9; i++)
		{// pragma unroll
			pix_neighborhood[i] = shared[tid.y][tid.x * 2 + i - 4 + OFFSET_97];
		}

		// Predict 1
		process_new<float, 1, 7> (a1, pix_neighborhood);

		// Update 1
		process_new<float, 2, 6> (a2, pix_neighborhood);

		// Predict 2
		process_new<float, 3, 5> (a3, pix_neighborhood);

		// Update 2
		process_new<float, 4, 4> (a4, pix_neighborhood);

		save_part_results_new<float, (MEMSIZE + (BLOCKSIZEY - 1)) / BLOCKSIZEY, 4, 5> (p_offset.y, results, pix_neighborhood);

		p_offset.y++;
		tid.y += BLOCKSIZEY;
	}

//	// Process columns
//	while (tid.y < p_size.x && tidx2 < p_size.y)
//	{
//		fprocess_97(tid.y, tidx2, p_offset.y, pix_neighborhood, shared, results);
//
//		p_offset.y++;
//		tid.y += BLOCKSIZEY;
//	}

	__syncthreads();

	tid.x = threadIdx.x;
	tid.y = threadIdx.y;

	// Row number
	p_offset.y = 0;
	// Row offset
	p_offset.x = (int) ceilf(p_size.x / 2.0f);

	// Safe results and rotate
	while (tid.y < p_size.y && tid.x < p_size.x)
	{
		// Can not dynamically index registers, avoid local memory usage.
		//		shared[tid.x][tid.y] = k2 * results[0 + p_offset.y * 2];
		//		if(tid.x + BLOCKSIZEX < p_size.y)
		//			shared[tid.x + BLOCKSIZEX][tid.y] = k1 * results[1 + p_offset.y * 2];
		save_to_shared2_new<float, (MEMSIZE + (BLOCKSIZEY - 1)) / BLOCKSIZEY> (k, tid, p_offset, p_size.x, results, shared);

		p_offset.y++;
		tid.y += BLOCKSIZEY;
	}
	__syncthreads();

	save_data_new<float>(tid, p_offset, p_size, img_size, step.x, odata, shared);
}

/**
 * @brief Computes forward wavelet transform 53.
 *
 * @param idata Input data.
 * @param odata Output data
 * @param img_size Struct with input image width and height.
 * @param step Struct with output image width and height.
 */
__global__
void fwt53_new(const float *idata, float *odata, const int2 img_size, const int2 step)
{
	/* Shared memory for part of the signal */
	__shared__ int shared[MEMSIZE][MEMSIZE + 1];

	/* Begining x, y of the PATCH */
	const int2 bid = make_int2(blockIdx.x * PATCHX, blockIdx.y * PATCHY);

	/* Thread id */
	short2 tid = make_short2(threadIdx.x, threadIdx.y);

	/* Threads offset to read margins */
	short2 p_offset;

	/* Compute patch size */
	const short2 p_size = make_short2(img_size.x - bid.x < PATCHX ? img_size.x - bid.x : PATCHX, img_size.y - bid.y < PATCHY ? img_size.y
			- bid.y : PATCHY);

	/* Allocate registers in order to compute even and odd pixels. */
	int pix_neighborhood[5];
	// Minimize registers usage. Right | bottom offset. Odd | even result pixels.
	int results[((MEMSIZE + (BLOCKSIZEY - 1)) / BLOCKSIZEY) * 2];

	read_data_new<int>(tid, bid, p_size, img_size, step.x, idata, shared, results, OFFSET_53);
	__syncthreads();

	// Thread x id
	tid.x = threadIdx.x;
	// Thread y id
	tid.y = threadIdx.y;

	// Row number
	p_offset.y = 0;

	// Process rows
	while (tid.y * 2 < p_size.y && tid.x < p_size.x + 2 * OFFSET_53)
	{
		#pragma unroll
		for (int i = 0; i < 5; i++)
		{
			pix_neighborhood[i] = shared[tid.y * 2 + i - 2 + OFFSET_53][tid.x];
		}

		pix_neighborhood[1] -= ((pix_neighborhood[0] + pix_neighborhood[2]) >> 1);
		pix_neighborhood[3] -= ((pix_neighborhood[2] + pix_neighborhood[4]) >> 1);
		// Update 1
	//	process53<int, 2, 2> (1, 2, 2, pix_neighborhood);
		pix_neighborhood[2] += ((pix_neighborhood[1] + pix_neighborhood[3] + 2) >> 2);

		save_part_results_new<int, (MEMSIZE + (BLOCKSIZEY - 1)) / BLOCKSIZEY, 2, 3> (p_offset.y, results, pix_neighborhood);

		p_offset.y++;
		tid.x += BLOCKSIZEY;
	}
	__syncthreads();

	tid.x = threadIdx.x;
	tid.y = threadIdx.y;
	p_offset.y = 0;
	// Column offset
	p_offset.x = (int) ceilf(p_size.y / 2.0f);

	// safe results and rotate
	while (tid.y < p_size.y && tid.x < p_size.x + 2 * OFFSET_53)
	{
		// Can not dynamically index registers, avoid local memory usage.
		//		shared[tid.x][tid.y] = k2 * results[0 + p_offset.y * 2];
		//		if(tid.x + BLOCKSIZEX < p_size.y)
		//			shared[tid.x + BLOCKSIZEX][tid.y] = k1 * results[1 + p_offset.y * 2];
		save_to_shared_new<int, (MEMSIZE + (BLOCKSIZEY - 1)) / BLOCKSIZEY> (1, tid, p_offset, p_size.y, results, shared);

		p_offset.y++;
		tid.x += BLOCKSIZEY;
	}
	__syncthreads();

	tid.x = threadIdx.x;
	tid.y = threadIdx.y;

	// Row number
	p_offset.y = 0;

	// Process columns
	while (tid.y < p_size.y && tid.x * 2 < p_size.x)
	{
		#pragma unroll
		for (int i = 0; i < 5; i++)
		{
			pix_neighborhood[i] = shared[tid.y][tid.x * 2 + i - 2 + OFFSET_53];
		}

		pix_neighborhood[1] -= ((pix_neighborhood[0] + pix_neighborhood[2]) >> 1);
		pix_neighborhood[3] -= ((pix_neighborhood[2] + pix_neighborhood[4]) >> 1);
		pix_neighborhood[2] += ((pix_neighborhood[1] + pix_neighborhood[3] + 2) >> 2);

		save_part_results_new<int, (MEMSIZE + (BLOCKSIZEY - 1)) / BLOCKSIZEY, 2, 3> (p_offset.y, results, pix_neighborhood);

		p_offset.y++;
		tid.y += BLOCKSIZEY;
	}

	__syncthreads();

	tid.x = threadIdx.x;
	tid.y = threadIdx.y;

	// Row number
	p_offset.y = 0;
	// Row offset
	p_offset.x = (int) ceilf(p_size.x / 2.0f);

	// Safe results and rotate
	while (tid.y < p_size.y && tid.x < p_size.x)
	{
		// Can not dynamically index registers, avoid local memory usage.
		//		shared[tid.x][tid.y] = k2 * results[0 + p_offset.y * 2];
		//		if(tid.x + BLOCKSIZEX < p_size.y)
		//			shared[tid.x + BLOCKSIZEX][tid.y] = k1 * results[1 + p_offset.y * 2];
		save_to_shared2_new<int, (MEMSIZE + (BLOCKSIZEY - 1)) / BLOCKSIZEY> (1, tid, p_offset, p_size.x, results, shared);

		p_offset.y++;
		tid.y += BLOCKSIZEY;
	}

	__syncthreads();

	save_data_new<int>(tid, p_offset, p_size, img_size, step.x, odata, shared);
}
