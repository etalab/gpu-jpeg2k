/*
 * @file common_worker.h
 *
 * @author Milosz Ciznicki 
 * @date 09-05-2011
 */

#ifndef COMMON_WORKER_H_
#define COMMON_WORKER_H_

void sleep_worker(hs_worker *worker);
void update_worker_exec_status(worker_timing *timing, struct timespec *ts_start, struct timespec *ts_end);

#endif /* COMMON_WORKER_H_ */
