/**
 * @file cost_model.h
 *
 * @author Milosz Ciznicki
 */

#ifndef COST_MODEL_H_
#define COST_MODEL_H_

#define HS_NARCH_IDS 2

enum hs_arch_ids {
	HS_CPU_ID = 0,
	HS_CUDA_ID = 1
};

typedef enum {
	HS_ARCH,
	HS_COMMON
} model_type;

typedef struct hs_arch_model_t {
	double (*task_cost)(void *cost_interface);
} hs_arch_model;

typedef struct hs_model_t {
	model_type type;

	/* single */
	hs_arch_model single_arch_model;

	/* per arch */
	hs_arch_model mutli_arch_model[HS_NARCH_IDS];
} hs_model;

#endif /* COST_MODEL_H_ */
