/**
 * @file scheduler.c
 *
 * @author Milosz Ciznicki
 */

#include "scheduler.h"
#include "../workers/worker.h"
#include "../policies/policy.h"
#include "../tasks/task.h"
#include "../timing/timing.h"

void init_scheduler()
{
	init_global_time();
	init_workers();
	init_sched_policy(&config);
	create_workers();
}

void shutdown_scheduler()
{
	shutdown_workers();
	deinit_sched_policy(&config);
}

hs_config *get_config()
{
	return &config;
}
