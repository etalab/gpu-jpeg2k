/**
 * @file gs.h
 *
 * @author Kamil Balwierz
 */

#ifndef GS_H_
#define GS_H_

#include "klt.h"

int gram_schmidt(int N, type_data* output, type_data *dinput, type_data* eValues, int J, type_data er);

#endif
