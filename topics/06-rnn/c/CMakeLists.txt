# Module 06: the bit-exact echo toy (full BPTT vs Elman's truncated twin), linking the shared nanograd lib.
add_executable(rnn rnn.c)
target_link_libraries(rnn PRIVATE nanograd)
