/**
 * @file test_cuda.c
 *
 * @author Milosz Ciznicki
 */

#include <time.h>
#include <errno.h>
#include "test.h"

void sleep_ms(unsigned int msec)
{
	struct timespec timeout0;
	struct timespec timeout1;
	struct timespec* tmp;
	struct timespec* t0 = &timeout0;
	struct timespec* t1 = &timeout1;

	t0->tv_sec = msec / 1000;
	t0->tv_nsec = (msec % 1000) * (1000 * 1000);

	while ((nanosleep(t0, t1) == (-1)) && (errno == EINTR)) {
		tmp = t0;
		t0 = t1;
		t1 = tmp;
	}
}

extern "C" void test_cuda_func(void *data_interface)
{
	data *data_i = (data *) data_interface;

	sleep_ms(data_i->gpu_time);
}
