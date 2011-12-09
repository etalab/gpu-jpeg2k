set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

find_package(PkgConfig REQUIRED)

cuda_add_library(mqc
	timer.c 
	mqc_data.c
	mqc_opj_helper.c
	mqc_configuration.cu
	mqc_develop.cu
	mqc_wrapper.cpp
	)

TARGET_LINK_LIBRARIES(mqc 
	misc
	openjpeg
)