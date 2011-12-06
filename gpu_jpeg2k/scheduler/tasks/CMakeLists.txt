set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS})

add_library(tasks 
task.c)

TARGET_LINK_LIBRARIES(tasks
policies)