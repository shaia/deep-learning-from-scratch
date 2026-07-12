# Module 03: the bit-exact toy CNN (bars), linking the shared nanograd lib.
add_executable(lenet lenet.c)
target_link_libraries(lenet PRIVATE nanograd)
