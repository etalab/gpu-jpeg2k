/*
 * print_info.h
 *
 *  Created on: 21-11-2010
 *      Author: Milosz Ciznicki
 */

#ifndef PRINT_INFO_H_
#define PRINT_INFO_H_

#define FUNC___ __FUNCTION__
#define FILE___ __FILE__
#define LINE___ __LINE__
#define INFO FILE___, FUNC___, LINE___

void println(const char* file, const char* function, const int line, const char *str);
void println_start(const char* file, const char* function, const int line);
void println_end(const char* file, const char* function, const int line);
void println_var(const char* file, const char* function, const int line, const char* format, ...);
//void start_measure();
long int start_measure();
long int stop_measure(long int start);
//long int stop_measure(const char* file, const char* function, const int line);
long int stop_measure_msg(const char* file, const char* function, const int line, char *msg);
long int stop_measure_no_info();

#endif /* PRINT_INFO_H_ */
