set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})


cuda_add_library(misc
cuda_errors.cu
memory_management.cu
)
