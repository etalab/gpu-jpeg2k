/*
 * @file rate-distortion.cu
 *
 * @author milosz 
 * @date 28-09-2010
 */
#include "rate-distortion.h"
#include <stdio.h>

/**
 * @brief Computes near optimal truncation lengths (R).
 *
 * @param mq
 * @param rates The buffer where is stored rate (coded length) at the end of each coding pass.
 * @param last_byte Last byte pushed out to the codestream.
 * @param out_buff The buffer with output bytes.
 * @param n Index in the array of the last terminated length.
 */
void near_optimal_truncation_length(mqe_t *mq, int rates[], int last_byte, int out_buff[], int n)
{
	/* Lower and upper bounds for the C and B registers*/
	int c_lower;
	int c_upper;
	int b_lower;
	int b_upper;
	int i;
	/* Current calculated length */
	int curr_length;
	/* Current byte */
	int curr_byte;

	/* For each coding pass */
	for(i = 0; i < n; i++)
	{
		/* Read MQ coder state */
		c_lower = mq->saved_mq_state[i].c;
		c_upper = mq->saved_mq_state[i].c + mq->saved_mq_state[i].a;
		b_lower = mq->saved_mq_state[i].b;
		b_upper = mq->saved_mq_state[i].b;

		/* Normalize to CT = 0 and deal with the carry bits */
		c_lower <<= mq->saved_mq_state[i].ct;
		c_upper <<= mq->saved_mq_state[i].ct;

		if(c_lower & 0x8000000)
		{
			b_lower++;
			c_lower &= 0x7FFFFFF;
		}

		if(c_upper & 0x8000000)
		{
			b_upper++;
			c_upper &= 0x7FFFFFF;
		}

		/* Set initial length and byte */
		curr_length = rates[i];
		curr_byte = last_byte;

		while(true)
		{
			/* Test for sufficiency of the current length */
			if((curr_byte == 0xFF) && (b_lower < 128) && (b_upper > 127))
			{
				/* We have enough bytes, we do not need the last byte 0xFF */
				curr_length--;
				break;
			}

			if((curr_byte != 0xFF) && (b_lower < 256) && (b_upper > 255))
			{
				/* We have enough bytes */
				break;
			}

			/* Retrieve the next byte from the codestream and update the bounds and byte count.
			 * The byte following the first Rk bytes */
			curr_byte = out_buff[curr_length];
			b_lower -= curr_byte;
			b_upper -= curr_byte;
			curr_length++;

			if(curr_byte == 0xFF)
			{
				b_lower <<= 7;
				b_lower |= (c_lower >> 20) & 0x7F;
				c_lower &= 0xFFFFF;
				c_lower <<= 7;
				b_upper <<= 7;
				b_upper |= (c_upper >> 20) & 0x7F;
				c_upper &= 0xFFFFF;
				c_upper <<= 7;
			} else
			{
				b_lower <<= 8;
				b_lower |= (c_lower >> 19) & 0xFF;
				c_lower &= 0x7FFFF;
				c_lower <<= 8;
				b_upper <<= 8;
				b_upper |= (c_upper >> 19) & 0xFF;
				c_upper &= 0x7FFFF;
				c_upper <<= 8;
			}
		}
		rates[i] = curr_length;
	}
}

/**
 * @brief Computes the valid truncation points that lie in a convex hull.
 *
 * @param rates Rates at each truncation point associated with coding passes.
 * @param dists The reduction in distortion for each truncation point.
 * @param n The number of truncation points in rates and dists arrays.
 */
void compute_convex_hull(int rates[], float dists[], int n)
{
	/* Last selected point */
	int p;
	/* Current point */
	int k;
	/* RD-slope for last selected point */
	float p_slope;
	/* RD-slope for current point */
	float k_slope;
	/* Rate difference */
	int delta_rate;
	/* Distortion difference */
	float delta_dist;
	/* Restart computation */
	int restart;

	p_slope = 0;

	do
	{
		restart = 0;
		p = -1;

		for(k = 1; k < n; k++)
		{
			/* Already invalidated point */
			if (rates[k] < 0)
			{
				continue;
			}

			if (p >= 0)
			{
				/* Calculate decrease in rate */
				delta_rate = rates[k]-rates[p];
				/* Calculate decrease in distortion */
				delta_dist = dists[k]-dists[p];
			} else
			{
				delta_rate = rates[k];
				delta_dist = dists[k];
			}

			/* If exactly same distortion don't eliminate if the rates are
			 * equal, otherwise it can lead to infinite slope in lossless
			 * coding. */
			if (delta_dist < 0 || (delta_dist == 0 && delta_rate > 0))
			{
				/* This point increases distortion => invalidate */
				rates[k] = -rates[k];
//				npnt--;
				/* Goto next point */
				continue;
			}
			k_slope = (float)(delta_dist/delta_rate);

			/* Check that there is a decrease in distortion, slope is not
			 * infinite (i.e. delta_dist is not 0) and slope is
			 * decreasing. */
			if (p>=0 && (delta_rate <= 0 || k_slope >= p_slope ))
			{
				/* Last point was not good
				 * Remove p from valid points*/
				rates[p] = -rates[p];
//				npnt--;
				/* Restart from the first one */
				restart = 1;
				break;
			} else
			{
				p_slope = k_slope;
				p = k;
			}
		}
	} while(restart);
}
