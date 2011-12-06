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

const float Wr = 0.299;
const float Wb = 0.114;
//const float Wg = 1 - Wr - Wb;
const float Wg = 1 - 0.299 - 0.114;
const float Umax = 0.436;
const float Vmax = 0.615;



#endif /* PREPROCESSING_CONSTANTS_CUH_ */
