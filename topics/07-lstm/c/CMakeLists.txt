# Module 07: the bit-exact latch toy (Module 06's RNN and three LSTM twins on the same batches), linking the shared nanograd lib.
add_executable(lstm lstm.c)
target_link_libraries(lstm PRIVATE nanograd)
