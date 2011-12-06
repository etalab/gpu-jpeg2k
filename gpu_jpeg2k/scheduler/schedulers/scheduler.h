/**
 * @file scheduler.h
 *
 * @author Milosz Ciznicki
 */

#ifndef SCHEDULER_H_
#define SCHEDULER_H_

#include <stdint.h>
#include <stdlib.h>
#include "../workers/worker.h"
#include "../models/model.h"

#define MAX_WORKERS 32

typedef struct hs_config_t {
	hs_worker workers[MAX_WORKERS];
	volatile int8_t running;
	int8_t nhcpus;
	int8_t nhgpus;
	int8_t ncpus;
	int8_t ngpus;
	int8_t nworkers;
} hs_config;

hs_config config;

void init_scheduler();
void shutdown_scheduler();
hs_config *get_config();

#endif /* SCHEDULER_H_ */
