/**
 * @file policy_helper.c
 *
 * @author Milosz Ciznicki
 */

#include "../workers/worker.h"
#include "policy_helper.h"

float get_worker_weight_based_on_speed(int arch)
{
	switch(arch)
	{
	case HS_ARCH_CPU: return CPU_WEIGHT_SPEED;
	case HS_ARCH_CUDA: return CUDA_WEIGHT_SPEED;
	}

	return 1.0f;
}
