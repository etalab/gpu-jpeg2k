#include "timer.h"
#include <stdlib.h>

void
timer_reset(struct timer_state* timer)
{
    timer->duration = 0.0;
}

void
timer_start(struct timer_state* timer)
{
    gettimeofday(&timer->startTime, NULL);
}

void
timer_pause(struct timer_state* timer)
{
    struct timeval endTime;
    long seconds, useconds;
        
    gettimeofday(&endTime, NULL);

    seconds  = endTime.tv_sec  - timer->startTime.tv_sec;
    useconds = endTime.tv_usec - timer->startTime.tv_usec;

    timer->duration += ((seconds) * 1000.0 + useconds/1000.0) + 0.5;
}

double
timer_stop(struct timer_state* timer)
{
    timer_pause(timer);
    double duration = timer->duration;
    timer_reset(timer);
    return duration;
}
