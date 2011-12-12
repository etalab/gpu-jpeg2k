#include "mqc_opj_helper.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "library/openjpeg/opj_convert.h"
#include "library/openjpeg/opj_includes.h"
#include "timer.h"

#define PXM_DFMT 10
#define PGX_DFMT 11
#define BMP_DFMT 12
#define RAW_DFMT 15

int get_file_format(char *filename) {
    unsigned int i;
    static const char *extension[] = {
        "pgx", "pnm", "pgm", "ppm", "bmp", "raw"
    };
    static const int format[] = {
        PGX_DFMT, PXM_DFMT, PXM_DFMT, PXM_DFMT, BMP_DFMT, RAW_DFMT
    };
    char * ext = strrchr(filename, '.');
    if (ext == NULL)
        return -1;   
    ext++;
    for ( i = 0; i < sizeof(format)/sizeof(*format); i++ ) {
        if ( strncasecmp(ext, extension[i], 3) == 0 ) {
            return format[i];
        }
    }
    return -1;
}


void
mqc_opj_helper_reset()
{
    mqc_set_callback_t1_begin(0, 0);
    mqc_set_callback_t1_end(0, 0);
    mqc_set_callback_cblk_begin(0, 0);
    mqc_set_callback_cblk_end(0, 0);
    mqc_set_callback_cxd_pair(0, 0);
    mqc_set_callback_cblk_bytes(0, 0);
    mqc_set_callback_renormalize(0, 0);
}

static unsigned int mqc_opj_helper_param_cblk_width = 64;
static unsigned int mqc_opj_helper_param_cblk_height = 64;
static unsigned int mqc_opj_helper_param_dwt = 1;
static bool mqc_opj_helper_param_irreversible = false;

void
mqc_opj_helper_parameters(unsigned int cblk_width, unsigned int cblk_height, int dwt, bool irreversible)
{
    mqc_opj_helper_param_cblk_width = cblk_width;
    mqc_opj_helper_param_cblk_height = cblk_height;
    mqc_opj_helper_param_dwt = dwt;
    mqc_opj_helper_param_irreversible = irreversible;
}

bool
mqc_opj_helper_encode(const char* filename, void(*callback_image_info)(opj_image_t*,void*), void* param)
{
    return mqc_opj_helper_encode_with_duration(filename, callback_image_info, param, 0);
}

bool
mqc_opj_helper_encode_with_duration(const char* filename, void(*callback_image_info)(opj_image_t*,void*), void* param, double* duration)
{
    // Setup parameters
    raw_cparameters_t raw_cp;
    raw_cp.rawWidth = 0;
    opj_cparameters_t parameters;
    opj_set_default_encoder_parameters(&parameters);
    strncpy(parameters.infile, filename, sizeof(parameters.infile)-1);
    
    // Determine format
    parameters.decod_format = get_file_format(parameters.infile);
    switch(parameters.decod_format) {
        case PGX_DFMT:
        case PXM_DFMT:
        case BMP_DFMT:
        case RAW_DFMT:
            break;
        default:
            printf("Unrecognized format for infile: %s [accept only *.pnm, *.pgm, *.ppm, *.pgx, *.bmp, *.raw]!\n",parameters.infile);
            return false;
    }

    // Load image    
    opj_image_t *image = NULL;
    switch (parameters.decod_format) {
    case PGX_DFMT:
        image = opj_pgx_to_image(parameters.infile, &parameters);
        break;
    case PXM_DFMT:
        image = opj_pnm_to_image(parameters.infile, &parameters);
        break;
    case BMP_DFMT:
        image = opj_bmp_to_image(parameters.infile, &parameters);
        break;
    case RAW_DFMT:
    {
        char signo;
        const char* s = "";
        if (sscanf(s, "%d,%d,%d,%d,%c", &raw_cp.rawWidth, &raw_cp.rawHeight, &raw_cp.rawComp, &raw_cp.rawBitDepth, &signo) == 5) {
            if (signo == 's') {
                raw_cp.rawSigned = true;
                printf("\nRaw file parameters: %d,%d,%d,%d Signed\n", raw_cp.rawWidth, raw_cp.rawHeight, raw_cp.rawComp, raw_cp.rawBitDepth); 
            }
            else if (signo == 'u') {
                raw_cp.rawSigned = false;
                printf("\nRaw file parameters: %d,%d,%d,%d Unsigned\n", raw_cp.rawWidth, raw_cp.rawHeight, raw_cp.rawComp, raw_cp.rawBitDepth);
            } 
            else {
                printf("\nError: invalid raw image parameters: Unknown sign of raw file\n");  
                printf("Please use the Format option -F:\n");
                printf("-F rawWidth,rawHeight,rawComp,rawBitDepth,s/u (Signed/Unsigned)\n");
                printf("Example: -i lena.raw -o lena.j2k -F 512,512,3,8,u\n");
                printf("Aborting\n");
            }
        }
        else {
            printf("\nError: invalid raw image parameters\n");
            printf("Please use the Format option -F:\n");
            printf("-F rawWidth,rawHeight,rawComp,rawBitDepth,s/u (Signed/Unsigned)\n");
            printf("Example: -i lena.raw -o lena.j2k -F 512,512,3,8,u\n");
            printf("Aborting\n");
            return 1;
        }
        image = opj_raw_to_image(parameters.infile, &parameters, &raw_cp);
        break;
    }
    }
    if ( image == NULL ) {
        printf("Can't load image!\n");
        return false;
    }

    // Call callback for image info
    if ( callback_image_info != 0 ) {
        callback_image_info(image,param);
    }

    // Setup default parameters 
    parameters.cod_format = 0;
    parameters.cp_disto_alloc = 1;	
    parameters.tcp_mct = image->numcomps == 3 ? 1 : 0;
    
    parameters.tcp_numlayers = 1;
    parameters.tcp_rates[0] = 0;

    parameters.cblockw_init = mqc_opj_helper_param_cblk_width;
    parameters.cblockh_init = mqc_opj_helper_param_cblk_height;
    parameters.numresolution = mqc_opj_helper_param_dwt;
    parameters.irreversible = mqc_opj_helper_param_irreversible ? 1 : 0;

    // Start timer
    struct timer_state timer_state;
    timer_reset(&timer_state);
    timer_start(&timer_state);

    // Encode
    opj_cinfo_t* cinfo = opj_create_compress(CODEC_J2K);
    opj_setup_encoder(cinfo, &parameters, image);
    opj_cio_t* cio = opj_cio_open((opj_common_ptr)cinfo, NULL, 0);
    bool bSuccess = opj_encode(cinfo,cio,image,parameters.index);
    if ( !bSuccess ) {
        opj_cio_close(cio);
        printf("Can't encode image!\n");
        return false;
    }

    // End timer
    if ( duration != 0 ) {
        *duration = timer_stop(&timer_state);
    }

    // Cleanup
    opj_cio_close(cio);
    opj_destroy_compress(cinfo);
    opj_image_destroy(image);

    return true;
}

