#include "mqc_data.h"
#include <stdio.h>
#include <stdlib.h>
#include "../../../my_common/my_common.h"
#include <string.h>
#include "mqc_opj_helper.h"

#define MQC_DATA_CBLK_CXDS_ALLOC_COUNT 1000
#define MQC_DATA_CBLK_LIST_ALLOC_COUNT 100

struct mqc_data_cblk*
mqc_data_cblk_create()
{
    struct mqc_data_cblk* cblk = (struct mqc_data_cblk*)my_malloc(sizeof(struct mqc_data_cblk));
    cblk->cxds = 0;
    cblk->cxd_count = 0;
    cblk->_cxd_alloc_count = 0;
    cblk->bytes = 0;
    cblk->byte_count = 0;
    cblk->w = 0;
    cblk->h = 0;
    cblk->magbits = 0;
    cblk->dwtLevel = 0;
    cblk->compType = 0;
    cblk->stepSize = 0.0;
    cblk->nominalWidth = 0;
    cblk->nominalHeight = 0;
    cblk->totalpasses = 0;
    return cblk;
}

struct mqc_data_cxd*
mqc_data_cblk_append(struct mqc_data_cblk* cblk, int cx, int d)
{
    cblk->cxd_count++;
    if ( cblk->cxd_count > cblk->_cxd_alloc_count ) {
        if ( cblk->_cxd_alloc_count == 0 ) {
            cblk->_cxd_alloc_count = MQC_DATA_CBLK_CXDS_ALLOC_COUNT;
            cblk->cxds = (struct mqc_data_cxd*)my_malloc(cblk->_cxd_alloc_count * sizeof(struct mqc_data_cxd));
            if ( cblk->cxds == 0 ) {
                printf("Failed to alloc CX,D array of size %d\n",cblk->_cxd_alloc_count);
                return 0;
            }
        }
        else {
            cblk->_cxd_alloc_count = cblk->_cxd_alloc_count + MQC_DATA_CBLK_CXDS_ALLOC_COUNT;
            cblk->cxds = realloc(cblk->cxds,cblk->_cxd_alloc_count * sizeof(struct mqc_data_cxd));
            if ( cblk->cxds == 0 ) {
                printf("Failed to realloc CX,D array to size %d\n",cblk->_cxd_alloc_count);
                return 0;
            }
        }
    }
    struct mqc_data_cxd* cxd = &cblk->cxds[cblk->cxd_count - 1];
    cxd->cx = cx;
    cxd->d = d;
    return cxd;
}

void
mqc_data_cblk_destroy(struct mqc_data_cblk* cblk)
{
    if ( cblk->cxds != 0 )
        free(cblk->cxds);
    if ( cblk->bytes != 0 )
        free(cblk->bytes);
    free(cblk);
}

struct mqc_data*
mqc_data_create()
{
    struct mqc_data* data = (struct mqc_data*)my_malloc(sizeof(struct mqc_data));
    data->filename[0] = '\0';
    data->number = 0;
    data->cblks = 0;
    data->cblk_count = 0;
    data->_cblk_alloc_count = 0;
    data->width = 0;
    data->height = 0;
    return data;
}

struct mqc_data_cblk*
mqc_data_append(struct mqc_data* data, struct mqc_data_cblk* cblk)
{
    data->cblk_count++;
    if ( data->cblk_count > data->_cblk_alloc_count ) {
        if ( data->_cblk_alloc_count == 0 ) {
            data->_cblk_alloc_count = MQC_DATA_CBLK_LIST_ALLOC_COUNT;
            data->cblks = (struct mqc_data_cblk**)my_malloc(data->_cblk_alloc_count * sizeof(struct mqc_data_cblk*));
            if ( data->cblks == 0 ) {
                printf("Failed to alloc CBLK array of size %d\n",data->_cblk_alloc_count);
                exit(1);
            }
        }
        else {
            data->_cblk_alloc_count = data->_cblk_alloc_count + MQC_DATA_CBLK_LIST_ALLOC_COUNT;
            data->cblks = (struct mqc_data_cblk**)realloc(data->cblks,data->_cblk_alloc_count * sizeof(struct mqc_data_cblk*));
            if ( data->cblks == 0 ) {
                printf("Failed to realloc CBLK array to size %d\n",data->_cblk_alloc_count);
                exit(1);
            }
        }
    }
    data->cblks[data->cblk_count - 1] = cblk;
    return cblk;
}

void
mqc_data_append_start_params(int w, int h, int *coeff_data, int magbits, int orient, int qmfbid, int level, double stepsize, struct mqc_data* data) {
	struct mqc_data_cblk* cblk = data->cblks[data->cblk_count - 1];
	cblk->w = w;
	cblk->h = h;
	cblk->nominalWidth = w;
	cblk->nominalHeight = h;
	cblk->coefficients = (int *) my_malloc(sizeof(int) * cblk->w * cblk->h);
	memcpy(cblk->coefficients, coeff_data, sizeof(int) * cblk->w * cblk->h);
	cblk->subband = orient; // ?
	cblk->dwtLevel = level;
	cblk->compType = qmfbid == 1 ? 0 : 1;
	cblk->stepSize = stepsize;
	cblk->magbits = magbits; //hardcoded
}

void
mqc_data_append_end_params(int totalpasses, struct mqc_data* data) {
	struct mqc_data_cblk* cblk = data->cblks[data->cblk_count - 1];
	cblk->totalpasses = totalpasses;
}

void
mqc_data_destroy(struct mqc_data* data)
{
    if ( data->cblks != 0 ) {
        int cblk;
        for ( cblk = 0; cblk < data->cblk_count; cblk++ )
            mqc_data_cblk_destroy(data->cblks[cblk]);
        free(data->cblks);
    }
    free(data);
}

/**
 * Callback for code-block begin
 */
void
mqc_data_on_cblk_begin(int w, int h, int *coeff_data, int magbits, int orient, int qmfbid, int level, double stepsize, void* param)
{
    struct mqc_data* data = (struct mqc_data*)param;
    struct mqc_data_cblk* cblk = mqc_data_cblk_create();
    mqc_data_append(data,cblk);
    mqc_data_append_start_params(w, h, coeff_data, magbits, orient, qmfbid, level, stepsize, data);
}

/**
 * Callback for code-block end
 */
void
mqc_data_on_cblk_end(int totalpasses, void* param)
{
    struct mqc_data* data = (struct mqc_data*)param;
    mqc_data_append_end_params(totalpasses, data);
}

/**
 * Callback for CX,D occurance
 */
void
mqc_data_on_cxd(int cx, int d, int mps, void* param)
{
    struct mqc_data* data = (struct mqc_data*)param;
    if ( data->cblk_count > 0 ) {
        struct mqc_data_cblk* cblk = data->cblks[data->cblk_count - 1];
        mqc_data_cblk_append(cblk,cx,d);
    }
}

/**
 * Callback for output byte occurance
 */
void
mqc_data_on_cblk_bytes(unsigned char* bytes, int byte_count, void* param)
{
    struct mqc_data* data = (struct mqc_data*)param;
    if ( data->cblk_count > 0 ) {
        struct mqc_data_cblk* cblk = data->cblks[data->cblk_count - 1];
        cblk->bytes = (unsigned char*)my_malloc(byte_count * sizeof(unsigned char));
        memcpy(cblk->bytes,bytes,byte_count * sizeof(unsigned char));
        cblk->byte_count = byte_count;
    }
}

/**
 * Callback for output byte occurance
 */
void
mqc_data_on_image_info(opj_image_t* image, void* param)
{
    struct mqc_data* data = (struct mqc_data*)param;
    data->width = image->x1;
    data->height = image->y1;
}

struct mqc_data*
mqc_data_create_from_image(const char* filename)
{
    // Create data
    struct mqc_data* data = mqc_data_create();

    // Set file name
    strcpy(data->filename,filename);

    // Init data callbacks
    mqc_opj_helper_reset();
    mqc_set_callback_cblk_begin(&mqc_data_on_cblk_begin, (void*)data);
    mqc_set_callback_cblk_end(&mqc_data_on_cblk_end, (void*)data);
    mqc_set_callback_cxd_pair(&mqc_data_on_cxd, (void*)data);
    mqc_set_callback_cblk_bytes(&mqc_data_on_cblk_bytes, (void*)data);

    // Encode with openjpeg
    bool result = mqc_opj_helper_encode(filename, &mqc_data_on_image_info, (void*)data);
    if ( result == false ) {
        mqc_data_destroy(data);
        return 0;
    }

    return data;
}

void
mqc_data_print_info(struct mqc_data* data)
{
    int index;

    // Resolution
    int width = data->width;
    int height = data->height;

    // Code-block count
    int cblk_count = data->cblk_count;

    // CX,D pair count
    int cxd_count = 0;
    for ( index = 0; index < data->cblk_count; index++ ) {
        cxd_count += data->cblks[index]->cxd_count;
    }

    // Byte count
    int byte_count = 0;
    for ( index = 0; index < data->cblk_count; index++ ) {
        byte_count += data->cblks[index]->byte_count;
    }

    // Print info
    printf("[MQ-Coder Data");
    if ( data->number != 0 )
        printf(" #%d",data->number);
    printf("]\n");
    if ( data->filename != 0 )
        printf("  Filename:           %s\n", data->filename);
    printf("  Resolution:         %dx%d\n", width, height);
    printf("  Code-block count:   %d\n", cblk_count);
    printf("  CX,D pair count:    %d\n", cxd_count);
    printf("  Byte count:         %d\n", byte_count);
}
