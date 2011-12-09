#include "mqc_configuration.h"
#include "mqc_common.h"
#include <iostream>
#include <cuda_runtime.h>

void
mqc_gpu_configuration_load(mqc_gpu_configuration & config, const char* configuration)
{
    if ( configuration == 0 )
        return;

    std::string cfg = configuration;
    if ( cfg == "" )
        return;

    int pos = (int)std::string::npos;

    // Load kernel
    config.kernel = 0;

    // Thread work and per count
    pos = cfg.find("N");
    if ( pos == std::string::npos )
        pos = -1;
    std::string tcfg = cfg;
    tcfg.erase(0,pos + 1);
    pos = tcfg.find("/");
    if ( pos != std::string::npos ) {
        config.threadWorkCount = atoi(tcfg.substr(0,pos).c_str());
        tcfg.erase(0,pos + 1);

        pos = tcfg.find("-");
        if ( pos == std::string::npos ) {
            config.threadPerCount = atoi(tcfg.c_str());
        } else {
            config.threadPerCount = atoi(tcfg.substr(0,pos).c_str());
            tcfg.erase(0,pos + 1);
        }
    }

    // CX,D load type
    pos = cfg.find("T");
    if ( pos != std::string::npos ) {
        int byte_count = atoi(cfg.substr(pos + 1,2).c_str());
        switch ( byte_count ) {
            case 1:
                config.cxdLoadType = mqc_gpu_configuration::Byte;
                break;
            case 2:
                config.cxdLoadType = mqc_gpu_configuration::Short;
                break;
            case 4:
                config.cxdLoadType = mqc_gpu_configuration::Integer;
                break;
            case 8:
                config.cxdLoadType = mqc_gpu_configuration::Double;
                break;
        }
    }

    // CX,D load count
    pos = cfg.find("L");
    if ( pos != std::string::npos ) {
        config.cxdLoadCount = atoi(cfg.substr(pos + 1,2).c_str());
    }
    
    // Calculate
    pos = cfg.find("C");
    if ( pos != std::string::npos ) {
        int calculate_count = atoi(cfg.substr(pos + 1,1).c_str());
        switch ( calculate_count ) {
            case 0:
                config.calculate = calculate_none;
                break;
            case 1:
                config.calculate = calculate_once;
                break;
            case 2:
                config.calculate = calculate_twice;
                break;
            case 3:
                config.calculate = calculate_tripple;
                break;
        }
    }
}

void
mqc_gpu_configuration_run(mqc_gpu_configuration & config, struct cxd_block* d_cxd_blocks, 
                          int cxd_block_count, unsigned char* d_cxds, unsigned char* d_bytes)
{
    if ( config.kernel == 0 )
        return;

    cudaError cuerr = cudaSuccess;

    int block_count = cxd_block_count / config.threadWorkCount + 1;

    dim3 dim_grid;
    dim_grid.x = block_count;
    if ( dim_grid.x > MAXIMUM_GRID_SIZE ) {
        dim_grid.x = MAXIMUM_GRID_SIZE;
        dim_grid.y = block_count / MAXIMUM_GRID_SIZE + 1;
    }
    dim3 dim_block(config.threadPerCount * config.threadWorkCount,1);

    config.kernel<<<dim_grid,dim_block>>>(d_cxd_blocks,cxd_block_count,d_cxds,d_bytes);
    cuerr = cudaThreadSynchronize();
    if ( cuerr != cudaSuccess ) {
        std::cerr << "Kernel encode failed: " << cudaGetErrorString(cuerr) << std::endl;
        return;
    }
}

