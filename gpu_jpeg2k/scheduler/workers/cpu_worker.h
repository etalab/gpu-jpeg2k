/**
 * @file cpu_worker.h
 *
 * @author Milosz Ciznicki
 */

#ifndef CPU_WORKER_H_
#define CPU_WORKER_H_

int get_ncpus();
void *cpu_worker(void *arg);

#endif /* CPU_WORKER_H_ */
