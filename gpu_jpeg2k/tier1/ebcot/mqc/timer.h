#ifndef _TIMER_H
#define _TIMER_H

#include <sys/time.h>

/**
 * Timer struct
 */
struct timer_state {
    struct timeval startTime;
    double duration;
};

/**
 * Reset timer
 */
void
timer_reset(struct timer_state* timer);

/**
 * Start measuring time
 */
void
timer_start(struct timer_state* timer);

/**
 * Pause measuring time
 */
void
timer_pause(struct timer_state* timer);

/**
 * Timer stop, return duration and reset
 */
double
timer_stop(struct timer_state* timer);

#endif
