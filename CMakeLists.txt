# Root build for the C side of "Deep Learning From Scratch".
#
# Standalone early topics (Modules 00-01) compile with a one-line clang command
# and are NOT built here. From Module 02 on, topics link the shared `nanograd`
# library; those are wired up below.
#
#   cmake -B build && cmake --build build
#   ./build/topics/02-training-toolkit/c/toolkit         (Windows: toolkit.exe)
cmake_minimum_required(VERSION 3.16)
project(deep_learning_from_scratch C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Warnings are bugs in teaching code (docs/conventions/c-style.md).
if(MSVC)
  add_compile_options(/W3)
else()
  add_compile_options(-Wall -Wextra)
endif()

add_subdirectory(lib/c/nanograd)

# One entry per module that links nanograd. Add new modules here as they land.
add_subdirectory(topics/02-training-toolkit/c)
