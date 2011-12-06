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
