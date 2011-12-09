#ifndef _MQC_CONFIGURATION_H
#define _MQC_CONFIGURATION_H

#ifndef DEFAULT_TWC
    #define DEFAULT_TWC 32
#endif

#ifndef DEFAULT_TPC
    #define DEFAULT_TPC 4
#endif

#ifndef DEFAULT_CLT
    #define DEFAULT_CLT double
    #define DEFAULT_CLT_ENUM mqc_gpu_configuration::Double
#endif

#ifndef DEFAULT_CLC
    #define DEFAULT_CLC 3
#endif

#ifndef DEFAULT_C
    #define DEFAULT_C calculate_once
#endif

typedef enum{
    calculate_none = 0,
    calculate_once = 1,
    calculate_twice = 2,
    calculate_tripple = 3
} calculate_t;

struct mqc_gpu_configuration {
    // Number of working threads per block
    unsigned int threadWorkCount;
    // Each working thread is in group of THREAD_PER_COUNT threads
    unsigned int threadPerCount;
    // Type for CX,D loading (int means 4 bytes at once)
    enum {Byte, Short, Integer, Double} cxdLoadType;
    // Load count for CX,D
    unsigned int cxdLoadCount;
    // Calculate
    calculate_t calculate;

    // Encode kernel type
    typedef void (*kernel_encode_t)(struct cxd_block*, int, unsigned char*, unsigned char*);
    // Selected kernel
    kernel_encode_t kernel;

    // Default configuration
    mqc_gpu_configuration()
    {
        reset();
    }

    void reset()
    {
        threadWorkCount = DEFAULT_TWC;
        threadPerCount = DEFAULT_TPC;
        cxdLoadType = DEFAULT_CLT_ENUM;
        cxdLoadCount = DEFAULT_CLC;
        calculate = DEFAULT_C;
        kernel = 0;
    }
};

void
mqc_gpu_configuration_load(mqc_gpu_configuration & config, const char* configuration);

void
mqc_gpu_configuration_run(mqc_gpu_configuration & config, struct cxd_block* d_cxd_blocks, 
                          int cxd_block_count, unsigned char* d_cxds, unsigned char* d_bytes);

#define MGCS_KERNEL(CONFIG,KERNEL,TWC,TPC,CLT,CLC,C) \
    CONFIG.kernel = KERNEL<TWC,TPC,CLT,CLC,C>; \

#define MGCS_NONE(CONFIG,KERNEL,TWC,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) \

#define MGCS_END(CONFIG,KERNEL,TWC,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) \
    MGCS_KERNEL(CONFIG,KERNEL,TWC,TPC,CLT,CLC,C) \

#define MGCS_ITERATE_TPC1(CONFIG,KERNEL,TWC,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) \
    case 1: M1(CONFIG,KERNEL,TWC,1,CLT,CLC,C,M2,M3,M4,M5,MGCS_NONE) break; \

#define MGCS_ITERATE_TPC2(CONFIG,KERNEL,TWC,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) \
    MGCS_ITERATE_TPC1(CONFIG,KERNEL,TWC,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) \
    case 2: M1(CONFIG,KERNEL,TWC,2,CLT,CLC,C,M2,M3,M4,M5,MGCS_NONE) break; \

#define MGCS_ITERATE_TPC16(CONFIG,KERNEL,TWC,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) \
    MGCS_ITERATE_TPC2(CONFIG,KERNEL,TWC,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) \
    case 4: M1(CONFIG,KERNEL,TWC,4,CLT,CLC,C,M2,M3,M4,M5,MGCS_NONE) break; \
    case 8: M1(CONFIG,KERNEL,TWC,8,CLT,CLC,C,M2,M3,M4,M5,MGCS_NONE) break; \
    case 16: M1(CONFIG,KERNEL,TWC,16,CLT,CLC,C,M2,M3,M4,M5,MGCS_NONE) break; \

#define MGCS_ITERATE_TPC32(CONFIG,KERNEL,TWC,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) \
    MGCS_ITERATE_TPC16(CONFIG,KERNEL,TWC,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) \
    case 32: M1(CONFIG,KERNEL,TWC,32,CLT,CLC,C,M2,M3,M4,M5,MGCS_NONE) break; \

#define MGCS_ITERATE_TPC_CAT(NAME,MAX,CONFIG,KERNEL,TWC,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) \
    NAME ## MAX (CONFIG,KERNEL,TWC,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) \

#define MGCS_ITERATE_TWC(CONFIG,KERNEL,TWC,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) \
    switch ( TWC ) { \
        case 1: switch ( TPC ) { case 32: M1(CONFIG,KERNEL,1,32,CLT,CLC,C,M2,M3,M4,M5,MGCS_NONE) break; } break; \
        case 2: switch ( TPC ) { MGCS_ITERATE_TPC_CAT(MGCS_ITERATE_TPC,32,CONFIG,KERNEL,2,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) } break; \
        case 4: switch ( TPC ) { MGCS_ITERATE_TPC_CAT(MGCS_ITERATE_TPC,32,CONFIG,KERNEL,4,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) } break; \
        case 8: switch ( TPC ) { MGCS_ITERATE_TPC_CAT(MGCS_ITERATE_TPC,32,CONFIG,KERNEL,8,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) } break; \
        case 16: switch ( TPC ) { MGCS_ITERATE_TPC_CAT(MGCS_ITERATE_TPC,32,CONFIG,KERNEL,16,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) } break; \
        case 32: switch ( TPC ) { MGCS_ITERATE_TPC_CAT(MGCS_ITERATE_TPC,32,CONFIG,KERNEL,32,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) } break; \
        case 64: switch ( TPC ) { MGCS_ITERATE_TPC_CAT(MGCS_ITERATE_TPC,32,CONFIG,KERNEL,64,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) } break; \
        case 128: switch ( TPC ) { MGCS_ITERATE_TPC_CAT(MGCS_ITERATE_TPC,32,CONFIG,KERNEL,128,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) } break; \
        case 256: switch ( TPC ) { MGCS_ITERATE_TPC_CAT(MGCS_ITERATE_TPC,16,CONFIG,KERNEL,256,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) } break; \
        case 512: switch ( TPC ) { MGCS_ITERATE_TPC_CAT(MGCS_ITERATE_TPC,2,CONFIG,KERNEL,512,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) } break; \
        default: break; \
    } \

#define MGCS_ITERATE_TWC_1024(CONFIG,KERNEL,TWC,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) \
    switch ( TWC ) { \
        case 1: switch ( TPC ) { case 32: M1(CONFIG,KERNEL,1,32,CLT,CLC,C,M2,M3,M4,M5,MGCS_NONE) break; } break; \
        case 2: switch ( TPC ) { MGCS_ITERATE_TPC_CAT(MGCS_ITERATE_TPC,32,CONFIG,KERNEL,2,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) } break; \
        case 4: switch ( TPC ) { MGCS_ITERATE_TPC_CAT(MGCS_ITERATE_TPC,32,CONFIG,KERNEL,4,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) } break; \
        case 8: switch ( TPC ) { MGCS_ITERATE_TPC_CAT(MGCS_ITERATE_TPC,32,CONFIG,KERNEL,8,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) } break; \
        case 16: switch ( TPC ) { MGCS_ITERATE_TPC_CAT(MGCS_ITERATE_TPC,32,CONFIG,KERNEL,16,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) } break; \
        case 32: switch ( TPC ) { MGCS_ITERATE_TPC_CAT(MGCS_ITERATE_TPC,32,CONFIG,KERNEL,32,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) } break; \
        case 64: switch ( TPC ) { MGCS_ITERATE_TPC_CAT(MGCS_ITERATE_TPC,32,CONFIG,KERNEL,64,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) } break; \
        case 128: switch ( TPC ) { MGCS_ITERATE_TPC_CAT(MGCS_ITERATE_TPC,32,CONFIG,KERNEL,128,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) } break; \
        case 256: switch ( TPC ) { MGCS_ITERATE_TPC_CAT(MGCS_ITERATE_TPC,16,CONFIG,KERNEL,256,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) } break; \
        case 512: switch ( TPC ) { MGCS_ITERATE_TPC_CAT(MGCS_ITERATE_TPC,2,CONFIG,KERNEL,512,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) } break; \
        case 1024: switch ( TPC ) { case 1: M1(CONFIG,KERNEL,1024,1,CLT,CLC,C,M2,M3,M4,M5,MGCS_NONE) break; } break; \
        default: break; \
    } \

#define MGCS_ITERATE_CLT(CONFIG,KERNEL,TWC,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) \
    switch ( CLT ) { \
        case mqc_gpu_configuration::Byte: M1(CONFIG,KERNEL,TWC,TPC,unsigned char,CLC,C,M2,M3,M4,M5,MGCS_NONE) break; \
        case mqc_gpu_configuration::Integer: M1(CONFIG,KERNEL,TWC,TPC,int,CLC,C,M2,M3,M4,M5,MGCS_NONE) break; \
        case mqc_gpu_configuration::Double: M1(CONFIG,KERNEL,TWC,TPC,double,CLC,C,M2,M3,M4,M5,MGCS_NONE) break; \
        default: break; \
    } \

#define MGCS_ITERATE_CLC(CONFIG,KERNEL,TWC,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) \
    switch ( CLC ) { \
        case 1: M1(CONFIG,KERNEL,TWC,TPC,CLT,1,C,M2,M3,M4,M5,MGCS_NONE) break; \
        case 2: M1(CONFIG,KERNEL,TWC,TPC,CLT,2,C,M2,M3,M4,M5,MGCS_NONE) break; \
        case 3: M1(CONFIG,KERNEL,TWC,TPC,CLT,3,C,M2,M3,M4,M5,MGCS_NONE) break; \
        case 4: M1(CONFIG,KERNEL,TWC,TPC,CLT,4,C,M2,M3,M4,M5,MGCS_NONE) break; \
        case 8: M1(CONFIG,KERNEL,TWC,TPC,CLT,8,C,M2,M3,M4,M5,MGCS_NONE) break; \
        case 16: M1(CONFIG,KERNEL,TWC,TPC,CLT,16,C,M2,M3,M4,M5,MGCS_NONE) break; \
        case 32: M1(CONFIG,KERNEL,TWC,TPC,CLT,32,C,M2,M3,M4,M5,MGCS_NONE) break; \
        default: break; \
    } \

#define MGCS_ITERATE_C(CONFIG,KERNEL,TWC,TPC,CLT,CLC,C,M1,M2,M3,M4,M5) \
    switch ( C ) { \
        case calculate_none: M1(CONFIG,KERNEL,TWC,TPC,CLT,CLC,calculate_none,M2,M3,M4,M5,MGCS_NONE) break; \
        case calculate_once: M1(CONFIG,KERNEL,TWC,TPC,CLT,CLC,calculate_once,M2,M3,M4,M5,MGCS_NONE) break; \
        case calculate_twice: M1(CONFIG,KERNEL,TWC,TPC,CLT,CLC,calculate_twice,M2,M3,M4,M5,MGCS_NONE) break; \
        case calculate_tripple: M1(CONFIG,KERNEL,TWC,TPC,CLT,CLC,calculate_tripple,M2,M3,M4,M5,MGCS_NONE) break; \
        default: break; \
    } \

#define MGCS_CHECK(CONFIG) \
    if ( CONFIG.kernel == 0 ) \
        std::cerr << "Wrong configuration!" << std::endl;

// Select by [calculate, cxdLoad]
#define mqc_gpu_configuration_select_CL(CONFIG, KERNEL) \
    CONFIG.threadWorkCount = DEFAULT_TWC; \
    CONFIG.threadPerCount = DEFAULT_TPC; \
    MGCS_ITERATE_C(CONFIG,KERNEL,DEFAULT_TWC,DEFAULT_TPC,CONFIG.cxdLoadType,DEFAULT_CLC,CONFIG.calculate,MGCS_ITERATE_CLT,MGCS_END,MGCS_NONE,MGCS_NONE,MGCS_NONE) \
    MGCS_CHECK(CONFIG) \

// Select by [calculate, threadCount]
#define mqc_gpu_configuration_select_CT(CONFIG, KERNEL) \
    MGCS_ITERATE_C(CONFIG,KERNEL,CONFIG.threadWorkCount,CONFIG.threadPerCount,DEFAULT_CLT,DEFAULT_CLC,CONFIG.calculate,MGCS_ITERATE_TWC,MGCS_END,MGCS_NONE,MGCS_NONE,MGCS_NONE) \
    MGCS_CHECK(CONFIG) \

// Select by [calculate, threadCount (with 1024/1)]
#define mqc_gpu_configuration_select_CT_1024(CONFIG, KERNEL) \
    MGCS_ITERATE_C(CONFIG,KERNEL,CONFIG.threadWorkCount,CONFIG.threadPerCount,DEFAULT_CLT,DEFAULT_CLC,CONFIG.calculate,MGCS_ITERATE_TWC_1024,MGCS_END,MGCS_NONE,MGCS_NONE,MGCS_NONE) \
    MGCS_CHECK(CONFIG) \

// Select by [calculate, cxdLoadType, threadCount]
#define mqc_gpu_configuration_select_CLT(CONFIG, KERNEL) \
    MGCS_ITERATE_C(CONFIG,KERNEL,CONFIG.threadWorkCount,CONFIG.threadPerCount,CONFIG.cxdLoadType,DEFAULT_CLC,CONFIG.calculate,MGCS_ITERATE_CLT,MGCS_ITERATE_TWC,MGCS_END,MGCS_NONE,MGCS_NONE) \
    MGCS_CHECK(CONFIG) \

// Select by [cxdLoadType, cxdLoadCount, threadCount]
#define mqc_gpu_configuration_select_LCT_1024(CONFIG, KERNEL) \
    MGCS_ITERATE_CLT(CONFIG,KERNEL,CONFIG.threadWorkCount,CONFIG.threadPerCount,CONFIG.cxdLoadType,CONFIG.cxdLoadCount,DEFAULT_C,MGCS_ITERATE_CLC,MGCS_ITERATE_TWC_1024,MGCS_END,MGCS_NONE,MGCS_NONE) \
    MGCS_CHECK(CONFIG) \

// Select by [threadCount]
#define mqc_gpu_configuration_select_T(CONFIG, KERNEL) \
    MGCS_ITERATE_TWC(CONFIG,KERNEL,CONFIG.threadWorkCount,CONFIG.threadPerCount,DEFAULT_CLT,DEFAULT_CLC,DEFAULT_C,MGCS_END,MGCS_NONE,MGCS_NONE,MGCS_NONE,MGCS_NONE) \
    MGCS_CHECK(CONFIG) \

#endif
