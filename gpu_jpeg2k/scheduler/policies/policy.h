/*
 * @file policy.h
 *
 * @author Milosz Ciznicki 
 * @date 05-05-2011
 */

#ifndef POLICY_H_
#define POLICY_H_

#include "../schedulers/scheduler.h"
#include "../queues/queue.h"

typedef struct hs_policy_t hs_policy;

struct hs_policy_t {
	void (*init)(hs_config *, hs_policy *);
	void (*deinit)(hs_config *, hs_policy *);
	void (*push_task)(hs_task_queue *, _task_t);
	_task_t (*pop_task)(hs_task_queue *);
	hs_task_queue *(*get_queue)();
	int nqueues;
};

void init_sched_policy(hs_config *config);
_task_t pop_task(hs_task_queue *task_queue);
void push_task(hs_task_queue *task_queue, _task_t task);
hs_policy *get_policy();
void deinit_sched_policy(hs_config *config);

#endif /* POLICY_H_ */
