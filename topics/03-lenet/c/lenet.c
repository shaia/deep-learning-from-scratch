/* ============================================================================
 * Module 03 — Convolutions / LeNet-5 (weight sharing + locality for images)
 *
 * The module that teaches nanograd to see: ng_conv2d_* and ng_maxpool2d_*
 * (lib/c/nanograd/conv.c) join the library, and this program is the smallest
 * complete CNN built from them -- conv(1->4, 3x3) -> ReLU -> maxpool 2x2 ->
 * linear(36->2) -- trained by Adam on the "bars" toy: 8x8 images containing
 * one bright vertical bar (class 0) or horizontal bar (class 1) over faint
 * noise. One oriented 3x3 kernel, slid everywhere, solves it; after training
 * the kernels ARE oriented edge detectors.
 *
 * Every op runs in the same order as the Python mirror (python/lenet.py,
 * class ToyCNN), drawing data and weights from the same 64-bit LCG streams,
 * so the C<->Python agreement test pins every number to a tight relative
 * tolerance (1e-9 gate; ~1e-15 observed) -- through convolution, pooling's
 * argmax routing, and 300 Adam steps.
 *
 * Note there is no "flatten" step anywhere below: the pooled feature maps
 * [n,F,3,3] are one flat row-major buffer, which is ALREADY the [n,36]
 * row-major matrix ng_linear_forward expects. In C, flatten is a no-op.
 *
 * Build (standalone, from repo root; Windows: omit -lm, math is in the CRT):
 *   clang -O2 -std=c11 -Wall -Wextra -I lib/c/nanograd \
 *     topics/03-lenet/c/lenet.c \
 *     lib/c/nanograd/rng.c lib/c/nanograd/nn.c lib/c/nanograd/init.c \
 *     lib/c/nanograd/optim.c lib/c/nanograd/conv.c -o lenet
 * Or via CMake:  cmake -B build && cmake --build build
 *
 * The full-size LeNet on MNIST lives in the Python mirror (vectorized via
 * im2col, checked by metric >= 0.98); this file is the bit-exact gate.
 *
 * Reference: docs/references/papers.md (Module 03) -- LeCun, Bottou, Bengio
 * & Haffner (1998), "Gradient-Based Learning Applied to Document Recognition".
 * ==========================================================================*/
#include "nanograd.h"

#include <stdio.h>

#define N         16          /* toy images                                  */
#define IMG       8           /* image height = width                        */
#define N_CH      1           /* input channels (grayscale)                  */
#define N_F       4           /* conv filters                                */
#define KS        3           /* kernel size                                 */
#define CONV_OUT  (IMG - KS + 1)        /* 6x6 feature maps (valid, stride 1) */
#define POOL      2                     /* pooling window and stride          */
#define POOL_OUT  (CONV_OUT / POOL)     /* 3x3 after max-pool                 */
#define N_FLAT    (N_F * POOL_OUT * POOL_OUT)   /* 36 inputs to the head      */
#define N_OUT     2

#define TOY_SEED  1           /* weight draws                                */
#define DATA_SEED 2           /* data draws (independent stream)             */
#define TOY_STEPS 300
#define ADAM_LR   0.01
#define ADAM_B1   0.9
#define ADAM_B2   0.999
#define ADAM_EPS  1e-8

/* The bars dataset, in the exact draw order python/lenet.py's make_bars()
 * uses. Per image: one uniform for the bar position, then 64 uniforms
 * (row-major) for the noise floor, then the bar overwritten at 1.0. Class is
 * i % 2 (no draw spent on it): even = vertical bar, odd = horizontal. */
static void make_bars(double X[N * N_CH * IMG * IMG], int y[N]) {
    NgRng rd;
    ng_rng_seed(&rd, DATA_SEED);
    for (int i = 0; i < N; i++) {
        y[i] = i % 2;
        int pos = (int)(ng_rng_uniform(&rd) * (double)IMG);
        double *img = X + i * IMG * IMG;
        for (int r = 0; r < IMG; r++)
            for (int c = 0; c < IMG; c++)
                img[r * IMG + c] = 0.25 * ng_rng_uniform(&rd);
        if (y[i] == 0)
            for (int r = 0; r < IMG; r++) img[r * IMG + pos] = 1.0;
        else
            for (int c = 0; c < IMG; c++) img[pos * IMG + c] = 1.0;
    }
}

static int argmax2(const double *p) { return p[1] > p[0] ? 1 : 0; }

int main(void) {
    /* Data (its own RNG stream, so data and weights can't entangle). */
    static double X[N * N_CH * IMG * IMG];
    int y[N];
    make_bars(X, y);

    /* Parameters. He init draws K1 flat then W2 row-major from one stream,
     * exactly as python/lenet.py's ToyCNN does. Biases start at zero. */
    NgRng rw;
    ng_rng_seed(&rw, TOY_SEED);
    double K1[N_F * N_CH * KS * KS], b1[N_F];
    double W2[N_FLAT * N_OUT], b2[N_OUT];
    ng_init_conv2d(K1, b1, N_F, N_CH, KS, KS, &rw, NG_INIT_HE);
    ng_init_linear(W2, b2, N_FLAT, N_OUT, &rw, NG_INIT_HE);

    /* Adam state: 1st/2nd moments per parameter, plus running powers of the
     * betas for bias-correction (advanced by multiplication, never pow()). */
    double mK1[N_F * N_CH * KS * KS] = {0}, vK1[N_F * N_CH * KS * KS] = {0};
    double mb1[N_F] = {0}, vb1[N_F] = {0};
    double mW2[N_FLAT * N_OUT] = {0}, vW2[N_FLAT * N_OUT] = {0};
    double mb2[N_OUT] = {0}, vb2[N_OUT] = {0};
    double b1_pow = 1.0, b2_pow = 1.0;

    /* Forward/backward scratch. POOLED[n,F,3,3] doubles as the [n,36] input
     * matrix of the linear head -- flatten is a no-op on a flat buffer. */
    static double Z[N * N_F * CONV_OUT * CONV_OUT];      /* conv pre-act    */
    static double A[N * N_F * CONV_OUT * CONV_OUT];      /* after ReLU      */
    static double POOLED[N * N_FLAT];
    static int    IDX[N * N_FLAT];                       /* pool winners    */
    static double LOGITS[N * N_OUT], P[N * N_OUT];
    static double DLOGITS[N * N_OUT], DPOOL[N * N_FLAT];
    static double DA[N * N_F * CONV_OUT * CONV_OUT];
    static double DZ[N * N_F * CONV_OUT * CONV_OUT];
    double gK1[N_F * N_CH * KS * KS], gb1[N_F];
    double gW2[N_FLAT * N_OUT], gb2[N_OUT];

    printf("[toy] conv(%d->%d, %dx%d) -> ReLU -> maxpool -> linear(%d->%d), "
           "Adam on bars\n", N_CH, N_F, KS, KS, N_FLAT, N_OUT);
    for (int t = 0; t < TOY_STEPS; t++) {
        /* Forward: conv -> ReLU -> max-pool -> (flatten: free) -> linear
         * -> softmax cross-entropy. */
        ng_conv2d_forward(X, K1, b1, Z, N, N_CH, IMG, IMG, N_F, KS, KS);
        ng_relu_forward(Z, A, N * N_F * CONV_OUT * CONV_OUT);
        ng_maxpool2d_forward(A, POOLED, IDX, N, N_F, CONV_OUT, CONV_OUT, POOL);
        ng_linear_forward(POOLED, W2, b2, LOGITS, N, N_FLAT, N_OUT);
        double loss = ng_softmax_ce_forward(LOGITS, y, P, N, N_OUT);

        /* Backward: the reversed walk. Max-pool routes each pooled gradient
         * to its window's winner; ReLU's mask then decides if it survives;
         * the conv backward turns what's left into dK1/db1 (dX not needed:
         * this conv touches the data directly). */
        ng_softmax_ce_backward(P, y, DLOGITS, N, N_OUT);
        ng_linear_backward(POOLED, W2, DLOGITS, DPOOL, gW2, gb2,
                           N, N_FLAT, N_OUT);
        ng_maxpool2d_backward(IDX, DPOOL, DA, N, N_F, CONV_OUT, CONV_OUT, POOL);
        ng_relu_backward(Z, DA, DZ, N * N_F * CONV_OUT * CONV_OUT);
        ng_conv2d_backward(X, K1, DZ, NULL, gK1, gb1,
                           N, N_CH, IMG, IMG, N_F, KS, KS);

        /* Adam update (advance bias-correction powers first: t := t+1). */
        b1_pow *= ADAM_B1;
        b2_pow *= ADAM_B2;
        double b1c = 1.0 - b1_pow, b2c = 1.0 - b2_pow;
        ng_adam_step(K1, gK1, mK1, vK1, N_F * N_CH * KS * KS,
                     ADAM_LR, ADAM_B1, ADAM_B2, ADAM_EPS, 0.0, b1c, b2c);
        ng_adam_step(b1, gb1, mb1, vb1, N_F,
                     ADAM_LR, ADAM_B1, ADAM_B2, ADAM_EPS, 0.0, b1c, b2c);
        ng_adam_step(W2, gW2, mW2, vW2, N_FLAT * N_OUT,
                     ADAM_LR, ADAM_B1, ADAM_B2, ADAM_EPS, 0.0, b1c, b2c);
        ng_adam_step(b2, gb2, mb2, vb2, N_OUT,
                     ADAM_LR, ADAM_B1, ADAM_B2, ADAM_EPS, 0.0, b1c, b2c);

        if (t % (TOY_STEPS / 10) == 0 || t == TOY_STEPS - 1)
            printf("  step %4d  loss=%.6f\n", t, loss);
    }

    /* Final forward pass for the fingerprint (loss/acc AFTER the last update,
     * matching ToyCNN.loss_and_acc in the Python mirror). */
    ng_conv2d_forward(X, K1, b1, Z, N, N_CH, IMG, IMG, N_F, KS, KS);
    ng_relu_forward(Z, A, N * N_F * CONV_OUT * CONV_OUT);
    ng_maxpool2d_forward(A, POOLED, IDX, N, N_F, CONV_OUT, CONV_OUT, POOL);
    ng_linear_forward(POOLED, W2, b2, LOGITS, N, N_FLAT, N_OUT);
    double loss = ng_softmax_ce_forward(LOGITS, y, P, N, N_OUT);
    int correct = 0;
    for (int i = 0; i < N; i++)
        if (argmax2(P + i * N_OUT) == y[i]) correct++;
    double acc = (double)correct / (double)N;

    /* wsum: a checksum over every parameter, summed in the same order as
     * Python (K1, b1, W2, b2). */
    double wsum = 0.0;
    for (int i = 0; i < N_F * N_CH * KS * KS; i++) wsum += K1[i];
    for (int i = 0; i < N_F; i++)                  wsum += b1[i];
    for (int i = 0; i < N_FLAT * N_OUT; i++)       wsum += W2[i];
    for (int i = 0; i < N_OUT; i++)                wsum += b2[i];

    printf("FINAL toy loss=%.17g acc=%.17g wsum=%.17g "
           "k1_0=%.17g w2_00=%.17g b2_0=%.17g\n",
           loss, acc, wsum, K1[0], W2[0], b2[0]);
    return 0;
}
