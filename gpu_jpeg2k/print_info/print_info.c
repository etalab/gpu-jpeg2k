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
/*
 * print_info.c
 *
 *  Created on: 21-11-2010
 *      Author: Milosz Ciznicki
 */
#include <stdio.h>
#include <stdarg.h>
#include <sys/time.h>
#include <time.h>

struct timeval start_time;

void println(const char* file, const char* function, const int line, const char *str)
{
	struct timeval tv;
	struct timezone tz;
	struct tm *tm;
	gettimeofday(&tv, &tz);
	tm = localtime(&tv.tv_sec);
	fprintf(stdout, "%d:%02d:%02d %6ld [%s] (%s:%d) %s\n", tm->tm_hour, tm->tm_min, tm->tm_sec, tv.tv_usec, function, file, line, str);
}

void println_var(const char* file, const char* function, const int line, const char* format, ...)
{
	char status[512];
	va_list arglist;

	va_start(arglist, format);
	vsprintf(status, format, arglist);
	va_end(arglist);

	println(file, function, line, status);
}

void println_start(const char* file, const char* function, const int line)
{
	println(file, function, line, "start");
}

void println_end(const char* file, const char* function, const int line)
{
	println(file, function, line, "end");
}

/*void start_measure()
{
	gettimeofday(&start_time, NULL);
}*/

long int start_measure()
{
	struct timeval start;
	gettimeofday(&start, NULL);

	return start.tv_sec * 1000000 + start.tv_usec;
}

long int stop_measure(long int start)
{
	struct timeval end;
	gettimeofday(&end, NULL);
	long int time = (end.tv_sec * 1000000 + end.tv_usec) - start;

	return time;
}

/*long int stop_measure(const char* file, const char* function, const int line)
{
	struct timeval end_time;
	gettimeofday(&end_time, NULL);
	long int time = (end_time.tv_sec - start_time.tv_sec) * 1000000 + end_time.tv_usec - start_time.tv_usec;
	println_var(file, function, line, "Computation time:%ld", time);

	return time;
}*/

long int stop_measure_msg(const char* file, const char* function, const int line, char *msg)
{
	struct timeval end_time;
	gettimeofday(&end_time, NULL);
	long int time = (end_time.tv_sec - start_time.tv_sec) * 1000000 + end_time.tv_usec - start_time.tv_usec;
	println_var(file, function, line, "%s:%ld", msg, time);

	return time;
}

long int stop_measure_no_info()
{
	struct timeval end_time;
	gettimeofday(&end_time, NULL);
	long int time = (end_time.tv_sec - start_time.tv_sec) * 1000000 + end_time.tv_usec - start_time.tv_usec;
	printf("%ld\n", time);

	return time;
}
