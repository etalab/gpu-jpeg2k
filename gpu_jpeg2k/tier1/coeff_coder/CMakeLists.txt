set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

find_package(PkgConfig REQUIRED)
#find_package(Boost REQUIRED)
#PKG_CHECK_MODULES(XML libxml++-2.6 REQUIRED)

#include_directories(${XML_INCLUDE_DIRS})

cuda_add_library(gpu_coeff_coder gpu_coeff_coder.cpp gpu_coeff_coder2.cu)
#target_link_libraries(gpu_coeff_coder ${XML_LIBRARIES})

TARGET_LINK_LIBRARIES(gpu_coeff_coder 
	misc
)