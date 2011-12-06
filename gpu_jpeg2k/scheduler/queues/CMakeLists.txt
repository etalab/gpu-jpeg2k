set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS})

add_library(queues 
deque_queue.c
fifo_queue.c
queue.c)

target_link_libraries(queues
    schedulers
)
