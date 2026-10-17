# Module 04: the bit-exact toy AlexNet (mirror glyphs), linking the shared nanograd lib.
add_executable(alexnet alexnet.c)
target_link_libraries(alexnet PRIVATE nanograd)
