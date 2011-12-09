#include "common.h"
#include "mqc_develop.h"
#include <iostream>
#include <stdint.h>

#define DEFAULT_TWC 64
#define DEFAULT_TPC 1
#define DEFAULT_CLC 8
#include "mqc_configuration.h"

// Context state
struct mqc_gpu_develop_state {
    // The probability of the Less Probable Symbol (0.75->0x8000, 1.5->0xffff)
    uint32_t qeval;
    // The Most Probable Symbol (0 or 1)
    uint8_t mps;
    // Number of shifts needed to be at least 0x8000
    uint8_t ns;
    // Next state index if the next encoded symbol is the MPS
    uint8_t nmps;
    // Next state index if the next encoded symbol is the LPS
    uint8_t nlps;
};

// Lookup table in GPU memory
__constant__ struct mqc_gpu_develop_state
d_mqc_gpu_develop_states[47 * 2];

// Lookup table in CPU memory
struct mqc_gpu_develop_state
mqc_gpu_develop_states[47 * 2] = {
    {0x5601, 0, 1,  2,  3},     {0x5601, 1, 1,  3,  2},
    {0x3401, 0, 2,  4,  12},    {0x3401, 1, 2,  5,  13},
    {0x1801, 0, 3,  6,  18},    {0x1801, 1, 3,  7,  19},
    {0x0ac1, 0, 4,  8,  24},    {0x0ac1, 1, 4,  9,  25},
    {0x0521, 0, 5,  10, 58},    {0x0521, 1, 5,  11, 59},
    {0x0221, 0, 6,  76, 66},    {0x0221, 1, 6,  77, 67},
    {0x5601, 0, 1,  14, 13},    {0x5601, 1, 1,  15, 12},
    {0x5401, 0, 1,  16, 28},    {0x5401, 1, 1,  17, 29},
    {0x4801, 0, 1,  18, 28},    {0x4801, 1, 1,  19, 29},
    {0x3801, 0, 2,  20, 28},    {0x3801, 1, 2,  21, 29},
    {0x3001, 0, 2,  22, 34},    {0x3001, 1, 2,  23, 35},
    {0x2401, 0, 2,  24, 36},    {0x2401, 1, 2,  25, 37},
    {0x1c01, 0, 3,  26, 40},    {0x1c01, 1, 3,  27, 41},
    {0x1601, 0, 3,  58, 42},    {0x1601, 1, 3,  59, 43},
    {0x5601, 0, 1,  30, 29},    {0x5601, 1, 1,  31, 28},
    {0x5401, 0, 1,  32, 28},    {0x5401, 1, 1,  33, 29},
    {0x5101, 0, 1,  34, 30},    {0x5101, 1, 1,  35, 31},
    {0x4801, 0, 1,  36, 32},    {0x4801, 1, 1,  37, 33},
    {0x3801, 0, 2,  38, 34},    {0x3801, 1, 2,  39, 35},
    {0x3401, 0, 2,  40, 36},    {0x3401, 1, 2,  41, 37},
    {0x3001, 0, 2,  42, 38},    {0x3001, 1, 2,  43, 39},
    {0x2801, 0, 2,  44, 38},    {0x2801, 1, 2,  45, 39},
    {0x2401, 0, 2,  46, 40},    {0x2401, 1, 2,  47, 41},
    {0x2201, 0, 2,  48, 42},    {0x2201, 1, 2,  49, 43},
    {0x1c01, 0, 3,  50, 44},    {0x1c01, 1, 3,  51, 45},
    {0x1801, 0, 3,  52, 46},    {0x1801, 1, 3,  53, 47},
    {0x1601, 0, 3,  54, 48},    {0x1601, 1, 3,  55, 49},
    {0x1401, 0, 3,  56, 50},    {0x1401, 1, 3,  57, 51},
    {0x1201, 0, 3,  58, 52},    {0x1201, 1, 3,  59, 53},
    {0x1101, 0, 3,  60, 54},    {0x1101, 1, 3,  61, 55},
    {0x0ac1, 0, 4,  62, 56},    {0x0ac1, 1, 4,  63, 57},
    {0x09c1, 0, 4,  64, 58},    {0x09c1, 1, 4,  65, 59},
    {0x08a1, 0, 4,  66, 60},    {0x08a1, 1, 4,  67, 61},
    {0x0521, 0, 5,  68, 62},    {0x0521, 1, 5,  69, 63},
    {0x0441, 0, 5,  70, 64},    {0x0441, 1, 5,  71, 65},
    {0x02a1, 0, 6,  72, 66},    {0x02a1, 1, 6,  73, 67},
    {0x0221, 0, 6,  74, 68},    {0x0221, 1, 6,  75, 69},
    {0x0141, 0, 7,  76, 70},    {0x0141, 1, 7,  77, 71},
    {0x0111, 0, 7,  78, 72},    {0x0111, 1, 7,  79, 73},
    {0x0085, 0, 8,  80, 74},    {0x0085, 1, 8,  81, 75},
    {0x0049, 0, 9,  82, 76},    {0x0049, 1, 9,  83, 77},
    {0x0025, 0, 10, 84, 78},    {0x0025, 1, 10, 85, 79},
    {0x0015, 0, 11, 86, 80},    {0x0015, 1, 11, 87, 81},
    {0x0009, 0, 12, 88, 82},    {0x0009, 1, 12, 89, 83},
    {0x0005, 0, 13, 90, 84},    {0x0005, 1, 13, 91, 85},
    {0x0001, 0, 15, 90, 86},    {0x0001, 1, 15, 91, 87},
    {0x5601, 0, 1,  92, 92},    {0x5601, 1, 1,  93, 93}
};

// Reset contexts
__device__ inline void
mqc_develop_reset_ctx(uint8_t* ctxs)
{
    ctxs[1]  = 0;  ctxs[9]  = 0;
    ctxs[2]  = 0;  ctxs[10] = 0;
    ctxs[3]  = 0;  ctxs[11] = 0;
    ctxs[4]  = 0;  ctxs[12] = 0;
    ctxs[5]  = 0;  ctxs[13] = 0;
    ctxs[6]  = 0;  ctxs[14] = 0;
    ctxs[7]  = 0;  ctxs[15] = 0;
    ctxs[8]  = 0;  ctxs[16] = 0;
    ctxs[0]  = (4 << 1);
    ctxs[17] = (3 << 1);
    ctxs[18] = (46 << 1);
}

// Perform byte out
__device__ inline void
mqc_develop_byte_out(uint32_t & c, uint8_t & ct, uint8_t* & bp)
{
    if ( *bp == 0xff ) {
        bp++;
        *bp = c >> 20;
        c &= 0xfffff;
        ct = 7;
    } else {
        if ( (c & 0x8000000) == 0 ) {
            bp++;
            *bp = c >> 19;
            c &= 0x7ffff;
            ct = 8;
        } else {
            (*bp)++;
            if ( *bp == 0xff ) {
                c &= 0x7ffffff;
                bp++;
                *bp = c >> 20;
                c &= 0xfffff;
                ct = 7;
            } else {
                bp++;
                *bp = c >> 19;
                c &= 0x7ffff;
                ct = 8;
            }
        }
    }
}

// Code MPS
__device__ inline void
mqc_develop_code_mps(uint32_t & a, uint32_t & c, uint8_t & ct,
                     uint8_t* & bp, uint8_t * & ctx, mqc_gpu_develop_state & state)
{
    int qeval =  state.qeval;
    a -= qeval;
    if ( (a & 0x8000) == 0 ) {
        if (a < qeval) {
            a = qeval;
        } else {
            c += qeval;
        }
        *ctx = state.nmps;

        a <<= 1;
        c <<= 1;
        ct--;
        if (ct == 0) {
            mqc_develop_byte_out(c,ct,bp);
        }
    } else {
        c += qeval;
    }
}

// Code LPS
__device__ inline void
mqc_develop_code_lps(uint32_t & a, uint32_t & c, uint8_t & ct,
                     uint8_t* & bp, uint8_t* & ctx, mqc_gpu_develop_state & state)
{
    int qeval =  state.qeval;
    a -= qeval;
    if ( a < qeval ) {
        c += qeval;

        *ctx = state.nlps;

        int ns = __clz(a) - (sizeof(uint32_t) * 8 - 16);

        a <<= ns;
        while ( ct <= ns ) {
            ns -= ct;
            c <<= ct;
            mqc_develop_byte_out(c,ct,bp);
        }
        c <<= ns;
        ct -= ns;
    } else {
        a = qeval;

        int ns = state.ns;

        *ctx = state.nlps;

        a <<= ns;
        while ( ct <= ns ) {
            ns -= ct;
            c <<= ct;
            mqc_develop_byte_out(c,ct,bp);
        }
        c <<= ns;
        ct -= ns;
    }
}

// Flush last bytes
__device__ inline void
mqc_develop_flush(uint32_t & a, uint32_t & c, uint8_t & ct, uint8_t* & bp)
{
    unsigned int tempc = c + a;
    c |= 0xffff;
    if ( c >= tempc ) {
        c -= 0x8000;
    }
    c <<= ct;
    mqc_develop_byte_out(c,ct,bp);
    c <<= ct;
    mqc_develop_byte_out(c,ct,bp);
    if ( *bp != 0xff ) {
        bp++;
    }
    c = c;
    ct = ct;
}

// Code symbol
__device__ inline void
mqc_develop_encode(uint8_t cxd, uint32_t & a, uint32_t & c, uint8_t & ct,
                   uint8_t* & bp, uint8_t * ctxs)
{
    uint8_t* ctx = &ctxs[cxd_get_cx(cxd)];
    mqc_gpu_develop_state state = d_mqc_gpu_develop_states[*ctx];
    if ( state.mps == cxd_get_d(cxd) ) {
        mqc_develop_code_mps(a,c,ct,bp,ctx,state);
    } else {
        mqc_develop_code_lps(a,c,ct,bp,ctx,state);
    }
}

// Kernel that performs MQ-Encoding for one block
template <unsigned int threadWorkCount, unsigned int threadPerCount, class cxdLoadType, unsigned int cxdLoadCount, calculate_t calculate>
__global__ void
kernel_mqc_gpu_develop_encode(struct cxd_block* d_cxd_blocks, int cxd_block_count, unsigned char* d_cxds, unsigned char* d_bytes)
{
    // Get and check block index
    int block_index = (blockIdx.y * gridDim.x + blockIdx.x) * threadWorkCount + threadIdx.x / threadPerCount;
    if ( block_index >= cxd_block_count )
        return;

    // Thread index in count
    int thread_index = threadIdx.x % threadPerCount;

    // Is this thread working
    bool work_thread = (thread_index) == 0;
    if ( work_thread == false )
        return;

    // Get block of CX,D pairs
    struct cxd_block* cxd_block = &d_cxd_blocks[block_index];

    // CX,D info
    int cxd_begin = cxd_block->cxd_begin;
    int cxd_count = cxd_begin + cxd_block->cxd_count;
    int cxd_index = cxd_begin;

    // Output byte stream
    uint8_t* start = &d_bytes[cxd_block->byte_begin];

    // Init variables
    uint32_t a = 0x8000;
    uint32_t c = 0;
    uint8_t ct = 12;
    uint8_t* bp = start - 1;
    uint8_t ctxs[19];
    if ( *bp == 0xff ) {
        ct = 13;
    }
    if ( calculate >= calculate_once ) {
        mqc_develop_reset_ctx(ctxs);
    }
    if ( calculate >= calculate_twice ) {
        mqc_develop_reset_ctx(ctxs);
    }
    if ( calculate >= calculate_tripple ) {
        mqc_develop_reset_ctx(ctxs);
    }
    
    // Get count of CX,D for align
    int align_count = cxd_index % sizeof(cxdLoadType);
    if ( align_count > 0 ) {
        // Make differ
        align_count = cxd_index + sizeof(cxdLoadType) - align_count;
        // Check count
        if ( align_count > cxd_count )
            align_count = cxd_count;
        // Encode align
        if ( calculate >= calculate_once ) {
            // Encode align symbols
            for ( cxd_index = cxd_index; cxd_index < align_count; cxd_index++ ) {
                uint8_t cxd = d_cxds[cxd_index];
                if ( calculate >= calculate_twice ) {
                    uint32_t a2 = a;
                    uint32_t c2 = c;
                    uint8_t ct2 = ct;
                    uint8_t* bp2 = bp;
                    mqc_develop_encode(cxd, a2, c2, ct2, bp2, ctxs);
                }
                if ( calculate >= calculate_tripple ) {
                    uint32_t a3 = a;
                    uint32_t c3 = c;
                    uint8_t ct3 = ct;
                    uint8_t* bp3 = bp;
                    mqc_develop_encode(cxd, a3, c3, ct3, bp3, ctxs);
                }
                mqc_develop_encode(cxd, a, c, ct, bp, ctxs);
            }
        }
    }
    
    while ( cxd_index < cxd_count ) {
        // Load CX,D by load type
        cxdLoadType cxd_data[cxdLoadCount];
        for ( int index = 0; index < cxdLoadCount; index++ )
            cxd_data[index] = reinterpret_cast<cxdLoadType*>(&d_cxds[cxd_index])[index];
               
        // Init count
        int count = sizeof(cxdLoadType) * cxdLoadCount;
        if ( (cxd_index + count) >= cxd_count ) {
            count = cxd_count - cxd_index;
        }

        // Encode CX,D
        for ( int index = 0; index < count; index++ ) {
            uint8_t cxd = reinterpret_cast<uint8_t*>(&cxd_data)[index];
            if ( calculate >= calculate_once ) {
                mqc_develop_encode(cxd, a, c, ct, bp, ctxs);
            }
            if ( calculate >= calculate_twice ) {
                uint32_t a2 = a;
                uint32_t c2 = c;
                uint8_t ct2 = ct;
                uint8_t* bp2 = bp;
                mqc_develop_encode(cxd, a2, c2, ct2, bp2, ctxs);
            }
            if ( calculate >= calculate_tripple ) {
                uint32_t a3 = a;
                uint32_t c3 = c;
                uint8_t ct3 = ct;
                uint8_t* bp3 = bp;
                mqc_develop_encode(cxd, a3, c3, ct3, bp3, ctxs);
            }
            if ( calculate == calculate_none) {
                a += cxd;
            }
        }
        
        cxd_index += count;
    }
    
    if ( calculate >= calculate_once ) {
        // Flush last bytes
        mqc_develop_flush(a,c,ct,bp); 

        // Set output byte count
        cxd_block->byte_count = bp - start;
    } else {
        *bp = a;
    }
}

// Configuration
mqc_gpu_configuration config_develop;

void
mqc_gpu_develop_init(const char* configuration)
{
    // Copy lookup table to constant memory
    cudaError cuerr = cudaMemcpyToSymbol(
        "d_mqc_gpu_develop_states",
        mqc_gpu_develop_states, 
        47 * 2 * sizeof(struct mqc_gpu_develop_state), 
        0, 
        cudaMemcpyHostToDevice
    );
    if ( cuerr != cudaSuccess ) {
        std::cerr << "Copy lookup table to constant failed: " << cudaGetErrorString(cuerr) << std::endl;
        return;
    }

    // Reset configuration
    config_develop.reset();

    // Load configuration
    mqc_gpu_configuration_load(config_develop, configuration);

    // Select kernel
    mqc_gpu_configuration_select_LCT_1024(config_develop, kernel_mqc_gpu_develop_encode);

    // Configure L1
    if ( config_develop.kernel != 0 )
        cudaFuncSetCacheConfig(config_develop.kernel,cudaFuncCachePreferL1);
}

// MQ-Encode input data
void
mqc_gpu_develop_encode(struct cxd_block* d_cxd_blocks, int cxd_block_count,
                       unsigned char* d_cxds, unsigned char* d_bytes)
{
    mqc_gpu_configuration_run(config_develop, d_cxd_blocks, cxd_block_count, d_cxds, d_bytes);
}

void
mqc_gpu_develop_deinit()
{
}

