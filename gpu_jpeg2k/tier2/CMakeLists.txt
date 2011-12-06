set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

cuda_add_library(tier2
codestream.c
codestream_mct.c
tag_tree_encode.c
write_codestream.c
buffer.c
)

TARGET_LINK_LIBRARIES(tier2
m
misc
types
tier1
)
