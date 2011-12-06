set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
#SET(CMAKE_CXX_FLAGS -g)
#SET(CMAKE_CC_FLAGS -g)

cuda_add_library(klt
	klt.cu
	gs.cu
	analysis.cu
	adjust.cu
	mct_transform.cu
	blocks.c
	reduce.cu
	mean_adjust_data.cu
	shift.cu
	calc_cov.cu
	calc_cov_mat.cu
)

target_link_libraries(klt
	m
)

cuda_add_cublas_to_target(klt)
