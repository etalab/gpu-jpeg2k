set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

cuda_add_library(workers
common_worker.c 
cpu_worker.c
gpu_worker.c
worker.c)

TARGET_LINK_LIBRARIES(workers
queues)