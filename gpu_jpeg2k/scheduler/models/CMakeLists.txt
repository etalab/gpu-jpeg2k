set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS})

add_library(models
model.c)

TARGET_LINK_LIBRARIES(models
policies)
