CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMake" "${PROJECT_SOURCE_DIR}/CMake/cuda" ${CMAKE_MODULE_PATH})
FIND_PACKAGE(CUDA)
FIND_PACKAGE(FreeImage)

#Uncomment next line to enable debugging (for use with cuda-gdb)
#SET(CUDA_NVCC_FLAGS -g -G --compiler-options -fpermissive)
SET(CMAKE_CXX_FLAGS "-O3")
SET(CMAKE_CC_FLAGS "-O3")
SET(CUDA_NVCC_FLAGS -O3 --compiler-options -fpermissive -arch sm_20 --maxrregcount 64 --ptxas-options=-v)
#SET(CUDA_NVCC_FLAGS --compiler-options -fpermissive -arch sm_20 --maxrregcount 64 --ptxas-options=-v)
SET(CUDA_VERBOSE_BUILD ON)

include_directories(${FREEIMAGE_INCLUDE_PATH})

add_subdirectory(tier1/coeff_coder)
add_subdirectory(tier1/bpc)
add_subdirectory(tier1/ebcot)
add_subdirectory(tier1/ebcot/mqc)
add_subdirectory(tier1/ebcot/mqc/library/openjpeg)
add_subdirectory(config)
add_subdirectory(dwt)
add_subdirectory(preprocessing)
add_subdirectory(misc)
add_subdirectory(tier1)
add_subdirectory(tier2)
add_subdirectory(file_format)
add_subdirectory(file_access)
add_subdirectory(types)
add_subdirectory(klt)

cuda_add_executable(encoder  
print_info/print_info.c
encoder.c)

TARGET_LINK_LIBRARIES( encoder 
${FREEIMAGE_LIBRARY} 
m
#gpu_coeff_coder 
ebcot
config
dwt
klt
preprocessing
misc
tier1
tier2
types)


cuda_add_executable(decoder  
print_info/print_info.c
decoder.c)

TARGET_LINK_LIBRARIES( decoder 
${FREEIMAGE_LIBRARY} 
m 
gpu_coeff_coder
config
dwt
preprocessing
misc
tier1
tier2
klt
file_format
file_access
types)

cuda_add_executable(jp2decoder  
print_info/print_info.c
jp2_decoder.c)

TARGET_LINK_LIBRARIES( jp2decoder
${FREEIMAGE_LIBRARY}
file_format
file_access
preprocessing
config
types
tier2
)

cuda_add_executable(test  
test.cu)

TARGET_LINK_LIBRARIES(test
${FREEIMAGE_LIBRARY}
file_format
file_access
preprocessing
config
types
tier1
ebcot
gpu_bpc
mqc
openjpeg
tier2
)
