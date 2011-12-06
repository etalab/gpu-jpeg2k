/*
 * @file model.h
 *
 * @author Milosz Ciznicki 
 * @date 19-05-2011
 */

#ifndef MODEL_H_
#define MODEL_H_

//#include "../schedulers/scheduler.h"
#include "../workers/worker.h"
#include "../tasks/task.h"
//#include "../queues/queue.h"

double calculate_task_length(hs_worker *worker_arg, _task_t _task);
hs_model *create_model();
void destroy_model(hs_model *model);

#endif /* MODEL_H_ */
