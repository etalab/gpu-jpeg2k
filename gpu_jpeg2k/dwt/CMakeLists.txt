set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

cuda_add_library(dwt 
dwt.cu
fwt.cu
fwt_new.cu
fwt_1d.cu
iwt_1d.cu
iwt.cu
iwt_new.cu
kernel.cu)
