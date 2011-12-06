/**
 * @file dequantizer.h
 *
 * @author Milosz Ciznicki
 */

#ifndef DEQUANTIZER_H_
#define DEQUANTIZER_H_

extern void dequantize_tile(type_tile *tile);
extern type_subband *dequantization(type_subband *sb);

#endif /* DEQUANTIZER_H_ */
