/*
 * @file vector_cpu_func.c
 *
 * @author Milosz Ciznicki 
 * @date 06-05-2011
 */

#include <time.h>
#include <errno.h>
#include "vector.h"

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

void scal_cpu_func(void *data_interface)
{
	data *data_i = (data *) data_interface;
	vector *vec = data_i->vec;
	float factor = data_i->factor;

	/* length of the vector */
	unsigned n = vec->size;
	float *h_array = vec->array;

	int i = 0;
	/* scale the vector */
	for (i = 0; i < n; i++)
		h_array[i] *= factor;

//	sleep_ms(2);
}
