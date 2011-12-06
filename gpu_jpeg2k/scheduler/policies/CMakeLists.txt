set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS})

add_library(policies
policy_helper.c 
policy.c
greedy_policy.c
weighted_rand_policy.c
work_stealing_policy.c
heft_policy.c)

TARGET_LINK_LIBRARIES(policies
models
queues)
