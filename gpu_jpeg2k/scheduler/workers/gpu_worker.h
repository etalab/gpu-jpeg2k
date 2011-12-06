/**
 * @file gpu_wokrer.h
 *
 * @author Milosz Ciznicki
 */

#ifndef GPU_WOKRER_H_
#define GPU_WOKRER_H_

int get_ngpus();
void *gpu_worker(void *arg);

#endif /* GPU_WOKRER_H_ */
