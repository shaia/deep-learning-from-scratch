/* ============================================================================
 * Module 02 — Making deep nets trainable (initialization, optimizers, regularization)
 *
 * The first topic to link the shared nanograd library (lib/c/nanograd) instead
 * of being one standalone file. It wires a 2->8->2 ReLU network with He
 * initialization and trains it on XOR with Adam -- the smallest program in which
 * the module's three new ideas (a non-saturating activation, fan-in-scaled init,
 * an adaptive optimizer) all appear together.
 *
 * Every op runs in the same order as the Python mirror (python/toolkit.py,
 * class ToyNet), drawing weights from the same 64-bit LCG, so the C<->Python
 * agreement test pins every number to full double precision -- even through
 * exp/log/sqrt and 2000 Adam steps.
 *
 * Build (standalone, from repo root; Windows: omit -lm, math is in the CRT):
 *   clang -O2 -std=c11 -Wall -Wextra -I lib/c/nanograd \
 *     topics/02-training-toolkit/c/toolkit.c \
 *     lib/c/nanograd/rng.c lib/c/nanograd/nn.c \
 *     lib/c/nanograd/init.c lib/c/nanograd/optim.c -o toolkit
 * Or via CMake:  cmake -B build && cmake --build build
 *
 * Reference: docs/references/papers.md (Module 02).
 * ==========================================================================*/
#include "nanograd.h"

#include <stdio.h>

#define N_IN      2
#define N_HID     8
#define N_OUT     2
#define N         4          /* the four XOR points */

#define TOY_SEED  1
#define TOY_STEPS 2000
#define ADAM_LR   0.05
#define ADAM_B1   0.9
#define ADAM_B2   0.999
#define ADAM_EPS  1e-8

/* XOR as a 2-class problem (labels are class indices for softmax cross-entropy). */
static const double TOY_X[N][N_IN] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
static const int    TOY_Y[N]       = { 0,      1,      1,      0    };

static int argmax2(const double *p) { return p[1] > p[0] ? 1 : 0; }

int main(void) {
    NgRng r;
    ng_rng_seed(&r, TOY_SEED);

    /* Parameters. He init draws W1 then W2 row-major from the same RNG stream,
     * exactly as python/toolkit.py's ToyNet does. Biases start at zero. */
    double W1[N_IN * N_HID], b1[N_HID];
    double W2[N_HID * N_OUT], b2[N_OUT];
    ng_init_linear(W1, b1, N_IN, N_HID, &r, NG_INIT_HE);
    ng_init_linear(W2, b2, N_HID, N_OUT, &r, NG_INIT_HE);

    /* Adam state: 1st/2nd moments per parameter, plus running powers of the
     * betas for bias-correction (advanced by multiplication, never pow()). */
    double mW1[N_IN * N_HID] = {0}, vW1[N_IN * N_HID] = {0};
    double mW2[N_HID * N_OUT] = {0}, vW2[N_HID * N_OUT] = {0};
    double mb1[N_HID] = {0}, vb1[N_HID] = {0};
    double mb2[N_OUT] = {0}, vb2[N_OUT] = {0};
    double b1_pow = 1.0, b2_pow = 1.0;

    /* Forward/backward scratch. */
    double z1[N * N_HID], a1[N * N_HID], logits[N * N_OUT], P[N * N_OUT];
    double dlogits[N * N_OUT], da1[N * N_HID], dz1[N * N_HID];
    double gW1[N_IN * N_HID], gb1[N_HID], gW2[N_HID * N_OUT], gb2[N_OUT];

    const double *X = &TOY_X[0][0];

    printf("[toy] 2->%d->2 ReLU, He init, Adam on XOR\n", N_HID);
    for (int t = 0; t < TOY_STEPS; t++) {
        /* Forward: linear -> ReLU -> linear -> softmax cross-entropy. */
        ng_linear_forward(X, W1, b1, z1, N, N_IN, N_HID);
        ng_relu_forward(z1, a1, N * N_HID);
        ng_linear_forward(a1, W2, b2, logits, N, N_HID, N_OUT);
        double loss = ng_softmax_ce_forward(logits, TOY_Y, P, N, N_OUT);

        /* Backward: the reversed walk through the same layers. */
        ng_softmax_ce_backward(P, TOY_Y, dlogits, N, N_OUT);
        ng_linear_backward(a1, W2, dlogits, da1, gW2, gb2, N, N_HID, N_OUT);
        ng_relu_backward(z1, da1, dz1, N * N_HID);
        ng_linear_backward(X, W1, dz1, NULL, gW1, gb1, N, N_IN, N_HID);

        /* Adam update (advance bias-correction powers first: t := t+1). */
        b1_pow *= ADAM_B1;
        b2_pow *= ADAM_B2;
        double b1c = 1.0 - b1_pow, b2c = 1.0 - b2_pow;
        ng_adam_step(W1, gW1, mW1, vW1, N_IN * N_HID,
                     ADAM_LR, ADAM_B1, ADAM_B2, ADAM_EPS, 0.0, b1c, b2c);
        ng_adam_step(W2, gW2, mW2, vW2, N_HID * N_OUT,
                     ADAM_LR, ADAM_B1, ADAM_B2, ADAM_EPS, 0.0, b1c, b2c);
        ng_adam_step(b1, gb1, mb1, vb1, N_HID,
                     ADAM_LR, ADAM_B1, ADAM_B2, ADAM_EPS, 0.0, b1c, b2c);
        ng_adam_step(b2, gb2, mb2, vb2, N_OUT,
                     ADAM_LR, ADAM_B1, ADAM_B2, ADAM_EPS, 0.0, b1c, b2c);

        if (t % (TOY_STEPS / 10) == 0 || t == TOY_STEPS - 1)
            printf("  step %5d  loss=%.6f\n", t, loss);
    }

    /* Final forward pass for the fingerprint (loss/acc AFTER the last update,
     * matching ToyNet.loss_and_acc in the Python mirror). */
    ng_linear_forward(X, W1, b1, z1, N, N_IN, N_HID);
    ng_relu_forward(z1, a1, N * N_HID);
    ng_linear_forward(a1, W2, b2, logits, N, N_HID, N_OUT);
    double loss = ng_softmax_ce_forward(logits, TOY_Y, P, N, N_OUT);
    int correct = 0;
    for (int i = 0; i < N; i++)
        if (argmax2(P + i * N_OUT) == TOY_Y[i]) correct++;
    double acc = (double)correct / (double)N;

    /* wsum: a checksum over every parameter, summed in the same order as Python. */
    double wsum = 0.0;
    for (int i = 0; i < N_IN * N_HID; i++)  wsum += W1[i];
    for (int i = 0; i < N_HID * N_OUT; i++) wsum += W2[i];
    for (int i = 0; i < N_HID; i++)         wsum += b1[i];
    for (int i = 0; i < N_OUT; i++)         wsum += b2[i];

    printf("FINAL toy loss=%.17g acc=%.17g wsum=%.17g "
           "w1_00=%.17g w2_00=%.17g b2_0=%.17g\n",
           loss, acc, wsum, W1[0], W2[0], b2[0]);
    return 0;
}
