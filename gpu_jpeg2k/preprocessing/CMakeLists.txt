set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

cuda_add_library(preprocessing
preprocess_gpu.cu
mct.c
)

target_link_libraries(preprocessing
	klt
)