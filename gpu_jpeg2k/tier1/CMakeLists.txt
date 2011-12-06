set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

cuda_add_library(tier1
quantization.c
quantizer.cu
dequantizer.cu
)
