set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

find_package(PkgConfig REQUIRED)

cuda_add_library(ebcot 
	gpu_coeff_coder.cpp 
	gpu_coeff_coder2.cu
	coeff_coder_pcrd.cu
	)

TARGET_LINK_LIBRARIES(ebcot 
	misc
	mqc
)