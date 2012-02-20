set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
#"--ptxas-options=-dlcm=cg"

cuda_add_library(gpu_bpc 
	bpc.cpp 
	gpu_bpc.cu
	gpu_convert.cu
	test_gpu_bpc.cpp
	)

TARGET_LINK_LIBRARIES(gpu_bpc 
	misc
)
