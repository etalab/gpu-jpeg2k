#ifndef _MQC_DATA_H
#define _MQC_DATA_H

/**
 * Struct that represents CX,D pair
 */
struct mqc_data_cxd {
    int cx;
    int d;
};

/**
 * Struct that represents code-block
 */
struct mqc_data_cblk {
    struct mqc_data_cxd* cxds;
    int cxd_count;
    int _cxd_alloc_count;
    int w;
    int h;
    int *coefficients;
    int subband;
    int nominalWidth;
    int nominalHeight;
    int magbits;
    int compType;
    int dwtLevel;
    int stepSize;
    unsigned char* bytes;
    int byte_count;
    int totalpasses;
};

/**
 * Struct that represents list of code-blocks
 */
struct mqc_data {
    char filename[255];
    int number;
    struct mqc_data_cblk** cblks;
    int cblk_count;
    int _cblk_alloc_count;
    int width;
    int height;
};

/**
 * Create code-block
 *
 * @return Returns new code-block pointer
 */
struct mqc_data_cblk*
mqc_data_cblk_create();

/**
 * Append CX,D pair to code-block
 *
 * @param cblk Pointer to code-block
 * @param cx Number of context
 * @param d Decision value
 */
struct mqc_data_cxd*
mqc_data_cblk_append(struct mqc_data_cblk* cblk, int cx, int d);

/**
 * Destroy code-block
 * 
 * @param cblk Pointer to code-block
 */
void
mqc_data_cblk_destroy(struct mqc_data_cblk* cblk);

/**
 * Create data
 *
 * @return Returns new data pointer
 */
struct mqc_data*
mqc_data_create();

/**
 * Append code-block to data
 *
 * @param data Pointer to data
 * @param cblk Pointer to code-block
 */
struct mqc_data_cblk*
mqc_data_append(struct mqc_data* data, struct mqc_data_cblk* cblk);

/**
 * Destroy data
 * 
 * @param data Pointer to data
 */
void
mqc_data_destroy(struct mqc_data* data);

/**
 * Create data from bmp image
 *
 * @param filename Filename of bmp image
 */
struct mqc_data*
mqc_data_create_from_image(const char* filename);

/**
 * Print info about data
 *
 * @param data Pointer to data
 */
void
mqc_data_print_info(struct mqc_data* data);

#endif
