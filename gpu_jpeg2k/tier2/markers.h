/**
 * @file markers.h
 *
 * @author Milosz Ciznicki
 */

#ifndef MARKERS_H_
#define MARKERS_H_

/* Markers */
/* Main header markers */
#define SOC 0xff4f
#define SIZ 0xff51
#define SSIZ_DEPTH_BITS 7
#define COD 0xff52
#define QCD 0xff5c
#define SQCX_GB_SHIFT 5
#define SQCX_EXP_SHIFT 3

/* Tile-part markers */
#define SOT 0xff90
#define SOD 0xff93

#define SOP 0xff91
#define EPH 0xff92
#define EOC 0xffd9

#define MCT 0xff74
#define MCC 0xff75
#define MIC 0xff78
#define ADS 0xff73
#define ATK 0xff72

#define CODING_STYLE 4

#define USED_SOP 0x02
#define USED_EPH 0x04

/* TODO: Compute number of layers */
#define NUM_LAYERS 1

/* Progression order */
#define LY_RES_COMP_POS_PROG 0
#define RES_LY_COMP_POS_PROG 1
#define RES_POS_COMP_LY_PROG 2
#define POS_COMP_RES_LY_PROG 3
#define COMP_POS_RES_LY_PROG 4

#endif /* MARKERS_H_ */
