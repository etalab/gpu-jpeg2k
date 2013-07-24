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
/**
 * @file quantization.c
 *
 * @author Milosz Ciznicki
 */
#include <math.h>
#include "quantization.h"

/**
 * @brief Converts from exponent-mantissa representation to floating-point. Use 11 least significant bits for mantissa and 5 bits for exponent.
 * @param ems
 * @return
 */
float convert_from_exp_mantissa(int ems)
{
	return (-1.0f - (float)(ems & 2047) / (float)(1 << 11)) / (float)(-1<<((ems>>11) & 31));
}

/**
 * @brief Converts floating-point to exponent-mantissa representation. Mantissa use 11 least significant bits and exponent 5 bits.
 * @param step
 * @return
 */
int convert_to_exp_mantissa(float step)
{
	/* TODO check if should be ceil or floor? */
	int exp = (int)floor(-log(step) / log(2));
	/* If overflow use minimum. */
	if(exp > 31)
	{
		return (31 << 11);
	}
	return (exp << 11) | ((int)((-step*(-1<<exp) - 1.0f)*(1<<11) + 0.5f));
}

/**
 * @brief Gets the base 2 exponent of the subband gain.
 * @param orient
 * @return
 */
int get_exp_subband_gain(int orient)
{
	return (orient & 1) + ((orient >> 1) & 1);
}
