set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

cuda_add_library(openjpeg
	bio.c 
	cio.c 
	dwt.c 
	event.c 
	image.c 
	j2k.c 
	j2k_lib.c 
	jp2.c 
	jpt.c 
	mct.c
    mqc.c 
    openjpeg.c 
    pi.c 
    raw.c 
    t1.c 
    t2.c 
    tcd.c 
    tgt.c 
    opj_convert.c
)

TARGET_LINK_LIBRARIES(openjpeg 

)