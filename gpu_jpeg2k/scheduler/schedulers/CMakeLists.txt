set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS})

add_library(schedulers 
scheduler.c)

TARGET_LINK_LIBRARIES(schedulers
workers)