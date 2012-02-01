/**
 * @brief Constants for the lossy color transformation.
 *
 * source: http://en.wikipedia.org/wiki/YUV#Conversion_to.2Ffrom_RGB
 *
 * @file preprocessing_constants.cuh
 *
 * @author Jakub Misiorny <misiorny@man.poznan.pl>
 */

#ifndef PREPROCESSING_CONSTANTS_CUH_
#define PREPROCESSING_CONSTANTS_CUH_

const float Wr = 0.299f;
const float Wb = 0.114f;
//const float Wg = 1 - Wr - Wb;
const float Wg = 1.0f - 0.299f - 0.114f;
const float Umax = 0.436f;
const float Vmax = 0.615f;



#endif /* PREPROCESSING_CONSTANTS_CUH_ */
