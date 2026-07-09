# Module 02's C program: a 2->8->2 ReLU MLP (He init, Adam) on XOR, built on the
# shared nanograd library. The first topic to link nanograd rather than being a
# single standalone file.
add_executable(toolkit toolkit.c)
target_link_libraries(toolkit PRIVATE nanograd)
