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
