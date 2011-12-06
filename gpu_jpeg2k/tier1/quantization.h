/**
 * @file quantization.h
 *
 * @author Milosz Ciznicki
 */

#ifndef QUANTIZATION_H_
#define QUANTIZATION_H_

#define BLOCKSIZEX 16
#define BLOCKSIZEY 16
#define COMPUTED_ELEMS_BY_THREAD 4

float convert_from_exp_mantissa(int ems);
int convert_to_exp_mantissa(float step);
int get_exp_subband_gain(int orient);

#endif /* QUANTIZATION_H_ */
