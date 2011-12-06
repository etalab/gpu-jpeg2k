/*
 * @file vector.h
 *
 * @author Milosz Ciznicki 
 * @date 06-05-2011
 */

#ifndef VECTOR_H_
#define VECTOR_H_

typedef struct vector_t {
	float *array;
	int size;
} vector;

typedef struct data_t {
	vector *vec;
	float factor;
} data;

#endif /* VECTOR_H_ */
