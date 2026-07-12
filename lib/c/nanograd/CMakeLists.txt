# The shared nanograd static library. Grows a source file at a time as later
# modules add layers/optimizers; consumers just link `nanograd`.
add_library(nanograd STATIC
  rng.c
  nn.c
  init.c
  optim.c
  conv.c
)

# Public header dir, so `#include "nanograd.h"` works for anything that links us.
target_include_directories(nanograd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# libm on POSIX toolchains; on Windows the math lives in the CRT.
if(NOT WIN32)
  target_link_libraries(nanograd PUBLIC m)
endif()
