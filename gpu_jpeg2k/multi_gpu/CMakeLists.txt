set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

cuda_add_library(multi_gpu
	entropy_coding_task.c
	entropy_coding.c
)

target_link_libraries(multi_gpu
	m
	misc
	schedulers
	tasks
	timing
	gpu_coeff_coder
)
