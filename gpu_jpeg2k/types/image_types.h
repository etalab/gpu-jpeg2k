/**
 * @file image_types.h
 *
 * @brief Defines all image/processing data structures.
 *
 * @author Milosz Ciznicki
 * @author Jakub Misiorny <misiorny@man.poznan.pl>
 */

#ifndef IMAGE_TYPES_H_
#define IMAGE_TYPES_H_

#include <stdint.h>
#include "../tier2/tag_tree_encode.h"
#include "image_mct.h"

#define UNSIGNED 0U
#define SIGNED 1U

#define DWT_53 0
#define DWT_97 1

#define GUARD_BITS 2

typedef struct type_coding_param type_coding_param;
typedef struct type_image type_image;
typedef struct type_tile type_tile;
typedef struct type_tile_comp type_tile_comp;
typedef struct type_res_lvl type_res_lvl;
typedef struct type_subband type_subband;
typedef struct type_codeblock type_codeblock;
typedef float type_data;

typedef enum
{
	LL, HL, LH, HH
} type_orient;

/** Codeblock coding parameters */
struct type_codeblock
{
	/** Codeblock number in raster order */
	uint32_t cblk_no;

	/** Codeblock number  in the horizontal direction */
	uint16_t no_x;

	/** Codeblock number  in the vertical direction */
	uint16_t no_y;

	/** The x-coordinate of the top-left corner of the codeblock, regarding to subband. */
	uint16_t tlx;

	/** The y-coordinate of the top-left corner of the codeblock, regarding to subband. */
	uint16_t tly;

	/** The x-coordinate of the bottom-right corner of the codeblock, regarding to subband. */
	uint16_t brx;

	/** The y-coordinate of the bottom-right corner of the codeblock, regarding to subband. */
	uint16_t bry;

	/** Codeblock width */
	uint16_t width;

	/** Codeblock height */
	uint16_t height;

	/** Pointer to codeblock data on gpu */
	int32_t *data_d;

	/** Parent subband */
	type_subband *parent_sb;

	/** Code block bytestream */
	uint8_t *codestream;

	/** Codestream length */
	uint32_t length;

	/** Significant bits in codeblock */
	uint8_t significant_bits;

	/** Number of length bits */
	uint32_t num_len_bits;

	/** Number of segments */
	uint32_t num_segments;

	/** Number of coding passes */
	uint32_t num_coding_passes;
};

/** Subband coding parameters */
struct type_subband
{
	/** The orientation of the subband(LL, HL, LH, HH) */
	type_orient orient;

	/** The x-coordinate of the top-left corner of the subband, regarding to tile-component. tbx0 */
	uint16_t tlx;

	/** The y-coordinate of the top-left corner of the subband, regarding to tile-component. tby0 */
	uint16_t tly;

	/** The x-coordinate of the bottom-right corner of the subband, regarding to tile-component. tbx1 */
	uint16_t brx;

	/** The y-coordinate of the bottom-right corner of the subband, regarding to tile-component. tby1 */
	uint16_t bry;

	/** Subband width */
	uint16_t width;

	/** Subband height */
	uint16_t height;

	/** Number of codeblocks in the horizontal direction in subband. */
	uint16_t num_xcblks;

	/** Number of codeblocks in the vertical direction in subband. */
	uint16_t num_ycblks;

	/** Total number of codeblocks in subband */
	uint32_t num_cblks;

	/** Codeblocks in current subband */
	type_codeblock *cblks;

	/** Codeblocks data on gpu */
	int32_t *cblks_data_d;

	/** Codeblocks data on cpu */
	int32_t *cblks_data_h;

	/** Number of magnitude bits in the integer representation of the quantized data */
	uint8_t mag_bits;

	/** Quantization step size */
	type_data step_size;

	/** Convert factor to quantize data */
	type_data convert_factor;

	/** Parent resolution-level */
	type_res_lvl *parent_res_lvl;

	/** Exponent */
	uint16_t expn;

	/** Matissa */
	uint16_t mant;

	/** Inclusion tag tree */
	type_tag_tree *inc_tt;
	/** Zero bit plane tag tree */
	type_tag_tree *zero_bit_plane_tt;
};

/** Resolution-level coding parameters */
struct type_res_lvl
{
	/** Resolution level number. r */
	uint8_t res_lvl_no;

	/** Decomposition level number. nb */
	uint8_t dec_lvl_no;

	/** The x-coordinate of the top-left corner of the tile-component
	 at this resolution. trx0 */
	uint16_t tlx;

	/** The y-coordinate of the top-left corner of the tile-component
	 at this resolution. try0 */
	uint16_t tly;

	/** The x-coordinate of the bottom-right corner of the tile-component
	 at this resolution(plus one). trx1 */
	uint16_t brx;

	/** The y-coordinate of the bottom-right corner of the tile-component
	 at this resolution(plus one). try1 */
	uint16_t bry;

	/** Resolution level width */
	uint16_t width;

	/** Resolution level height */
	uint16_t height;

	/** The exponent value for the precinct width. PPx */
	uint8_t prc_exp_w;

	/** The exponent value for the precinct height PPy */
	uint8_t prc_exp_h;

	/** Number of precincts in the horizontal direction in resolution level. numprecinctswide */
	uint16_t num_hprc;

	/** Number of precincts in the vertical direction in resolution level. numprecinctshigh */
	uint16_t num_vprc;

	/** Total number of precincts. numprecincts */
	uint16_t num_prcs;

	/** Number of subbands */
	uint8_t num_subbands;

	/** Subbands in current resolution level */
	type_subband *subbands;

	/** Parent tile-component */
	type_tile_comp *parent_tile_comp;
};

/** Tile on specific component/channel */
struct type_tile_comp
{
	/** Tile_comp number */
	uint32_t tile_comp_no;

	/** XXX: Tiles on specific components may have different sizes, because components can have various sizes. See ISO B.3 */
	/** Tile-component width. */
	uint16_t width;

	/** Tile-component height */
	uint16_t height;

	/** Number of decomposition levels. NL. COD marker */
	uint8_t num_dlvls;

	/** Number of the resolution levels. */
	uint8_t num_rlvls;

	/** The max exponent value for code-block width */
	/** XXX: Minimum for code-block dimension is 4.
	 * 	Maximum dimension is 64.  */
	uint8_t cblk_exp_w;

	/** The max exponent value for code-block height */
	uint8_t cblk_exp_h;

	/** Nominal codeblock width */
	uint8_t cblk_w;

	/** Nominal codeblock height */
	uint8_t cblk_h;

	/** Quantization style for all components */
	uint16_t quant_style;

	/** Number of guard bits */
	uint8_t num_guard_bits;

	/** Tile component data in the host memory (this is page-locked memory, prepared for copying to device) */
	type_data *img_data;

	/** Tile component data on the GPU */
	type_data *img_data_d;

	/** Resolution levels */
	type_res_lvl *res_lvls;

	/** Parent tile */
	type_tile *parent_tile;
};

struct type_tile
{
	/** Tile number in raster order */
	uint32_t tile_no;

	/** The x-coord of the top left corner of the tile with respect to the original image. tx0 */
	uint16_t tlx;

	/** The y-coord of the top left corner of the tile with respect to the original image. ty0 */
	uint16_t tly;

	/** The x-coord of the bottom right corner of the tile with respect to the original image. tx1 */
	uint16_t brx;

	/** The y-coord of the bottom right corner of the tile with respect to the original image. ty1 */
	uint16_t bry;

	/** Tile width */
	uint16_t width;

	/** Tile height */
	uint16_t height;

	/** Quantization style for each channel (ready for QCD/QCC marker) */
	int8_t QS;

	/** Tile on specific component/channel in host memory */
	type_tile_comp *tile_comp;

	/** Parent image */
	type_image *parent_img;
};

/* Coding parameters */
struct type_coding_param
{
	/* Image area */
	/** The horizontal offset from the origin of the reference grid to the
	 left edge of the image area. XOsiz */
	uint16_t imgarea_tlx;
	/** The vertical offset from the origin of the reference grid to the
	 left edge of the image area. YOsiz */
	uint16_t imgarea_tly;

	/** The horizontal offset from the origin of the reference grid to the
	 right edge of the image area. Xsiz */
	uint16_t imgarea_width;
	/** The vertical offset from the origin of the reference grid to the
	 right edge of the image area. Ysiz */
	uint16_t imgarea_height;

	/* Tile grid */
	/** The horizontal offset from the origin of the tile grid to the
	 origin of the reference grid. XTOsiz */
	uint16_t tilegrid_tlx;
	/** The vertical offset from the origin of the tile grid to the
	 origin of the reference grid. YTOsiz */
	uint16_t tilegrid_tly;

	/** The component horizontal sampling factor. XRsiz */
	uint16_t comp_step_x;

	/** The component vertical sampling factor. YRsiz */
	uint16_t comp_step_y;

	/** Base step size */
	type_data base_step;

	/** Target size when using PCRD */
	uint32_t target_size;
};

struct type_image
{
	/** Input file name */
	const char *in_file;

	/** Input header file name */
	const char *in_hfile;

	/** Output file name */
	const char *out_file;

	/** Configuration file */
	const char *conf_file;

	/** Mct_compression_method: 0 klt, 2 wavelet */
	uint8_t mct_compression_method;

	/** BSQ file type */
	uint8_t bsq_file;

	/** BIP file type */
	uint8_t bip_file;

	/** BIL file type */
	uint8_t bil_file;

	/** Image width */
	uint16_t width;

	/** Image height */
	uint16_t height;

	/** Number of channels/components. Csiz */
	uint16_t num_components;

	/** Original bit depth. XXX: Should be separate for every component */
	uint16_t depth;

	/** Data sign. XXX: Should be separate for every component.
	 * Really separate? i think we can safely assume all components are either signed or unsigned (q)*/
	uint8_t sign;

	/** Nominal number of decomposition levels */
	uint8_t num_dlvls;

	/** Type of wavelet transform: lossless DWT_53, lossy DWT_97. COD marker */
	uint8_t wavelet_type;

	/** Area allocated in the device memory */
	int32_t area_alloc;

	/** The nominal tile width. XTsiz. SIZ marker */
	uint16_t tile_w;
	/** The nominal tile height. YTsiz. SIZ marker */
	uint16_t tile_h;

	/** Number of tiles in horizontal direction. nunXtiles */
	uint16_t num_xtiles;
	/** Number of tiles in vertical direction. numYtiles */
	uint16_t num_ytiles;
	/** Number of all tiles */
	uint32_t num_tiles;

	/** Was the MCT used? */
	uint8_t use_mct;

	/** Was MCT as in 15444-2 standard */
	uint8_t use_part2_mct;

	/** Data for MCT as in 15444-2 */
	type_multiple_component_transformations* mct_data;

	/** Nominal range of bits */
	uint8_t num_range_bits;

	/** Coding style for all components */
	int32_t coding_style;

	/** Codeblock coding style */
	uint8_t cblk_coding_style;

	/** Progression order */
	uint8_t prog_order;

	/** Number of layers */
	uint16_t num_layers;

	/** Initial real-image data on GPU, used only in read_image and color transformation,
	 * after tiling use pointers in tile->tile_comp.*/
	type_data *img_data_d;
	/** Real image data is in this array of tiles. */
	type_tile *tile;

	/** Coding parameters */
	type_coding_param *coding_param;
};

#endif
