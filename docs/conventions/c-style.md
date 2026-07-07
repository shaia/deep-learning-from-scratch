# C Style Guide

C is our "no magic" language: no autograd, no hidden broadcasting, just arrays and arithmetic.
The reader should be able to trace every number. Clarity beats cleverness and beats speed.

## Language & dependencies

- **C11**, standard library + **`libm` only**. No third-party deps in teaching code.
- Compile clean under `-std=c11 -Wall -Wextra`. Warnings are bugs.
- Standalone early modules (00–01) are a **single `.c` file** readable top-to-bottom.
  From Module 02 on, topics link against `lib/c/nanograd`.

## Layout of a teaching `.c` file

Read like a lesson, in this order:

1. **File header comment** — what idea this is, the one-line intuition, the reference paper.
2. Includes, then small config `#define`s / `enum`s.
3. **Data**: how the toy dataset is generated (with a fixed seed).
4. **Model**: the forward pass, commented with the math it implements.
5. **Learning**: the gradient / update rule, **written out explicitly** — the reader sees the
   chain rule, not a call into a solver.
6. **Train loop** with periodic prints (epoch, loss, accuracy).
7. `main()` — wire it together, print final result.

## Naming

- Match `math-notation.md`: `w` weights, `b` bias, `x` input, `y` target, `yhat` prediction,
  `lr` learning rate, `grad_w` for ∂L/∂w, etc.
- Snake_case for functions and variables; `UPPER_CASE` for `#define` constants.
- Dimensions explicit in names when helpful: `n_samples`, `n_features`, `n_hidden`.

## Numerics & memory

- `double` by default in teaching code (fewer surprises); note where `float` would be used at scale.
- **Deterministic**: seed the RNG explicitly (a small `xorshift`/LCG is fine and keeps us
  dependency-free) so C and Python can be aligned for the agreement test.
- Prefer stack arrays / fixed sizes in tiny demos. When heap is needed, every `malloc` has a
  matching `free`; check for `NULL`.
- No function-static mutable state (global rule) — pass state via structs and parameters.

## Comments

- **High density in teaching files.** Explain *why*, and name the math each block implements
  (e.g. `// ∂L/∂w_j = (yhat - y) * x_j  — gradient of BCE through the sigmoid`).
- A short comment above each function stating its contract (inputs, output, shapes).

## Build

- Standalone: `clang -O2 -std=c11 <file>.c -o <name> -lm`
- Library-linked: via root `CMakeLists.txt` (`cmake -B build && cmake --build build`).

## Anti-patterns to avoid

- Clever pointer arithmetic that obscures the indexing math.
- Fusing steps to save lines when separate steps read clearer.
- Premature optimization — correctness and legibility first; note the fast version in prose.
