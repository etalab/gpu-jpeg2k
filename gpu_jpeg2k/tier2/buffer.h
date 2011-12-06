/*
 * buffer.h
 *
 *  Created on: Dec 1, 2011
 *      Author: miloszc
 */

#ifndef BUFFER_H_
#define BUFFER_H_

#include <stdio.h>
#include "../types/buffered_stream.h"

void init_dec_buffer(FILE *fsrc, type_buffer *src_buff);

#endif /* BUFFER_H_ */
