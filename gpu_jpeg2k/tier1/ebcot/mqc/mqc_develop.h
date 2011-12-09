#ifndef _MQC_GPU_DEVELOP_H
#define _MQC_GPU_DEVELOP_H

#include "mqc_common.h"

void
mqc_gpu_develop_init(const char* param);

void
mqc_gpu_develop_encode(struct cxd_block* d_cxd_blocks, int cxd_block_count, unsigned char* d_cxds, unsigned char* d_bytes);

void
mqc_gpu_develop_deinit();

#endif
