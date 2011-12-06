/**
 * @file policy_helper.h
 *
 * @author Milosz Ciznicki
 */

#ifndef POLICY_HELPER_H_
#define POLICY_HELPER_H_

#define CPU_WEIGHT_SPEED 1.0f
#define CUDA_WEIGHT_SPEED 10.0f

float get_worker_weight_based_on_speed(int arch);

#endif /* POLICY_HELPER_H_ */
