/* 
Copyright 2009-2013 Poznan Supercomputing and Networking Center

Authors:
Milosz Ciznicki miloszc@man.poznan.pl

GPU JPEG2K is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GPU JPEG2K is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with GPU JPEG2K. If not, see <http://www.gnu.org/licenses/>.
*/
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
