# Module 05: the bit-exact toy ResNet (deep spirals, plain vs residual twins), linking the shared nanograd lib.
add_executable(resnet resnet.c)
target_link_libraries(resnet PRIVATE nanograd)
