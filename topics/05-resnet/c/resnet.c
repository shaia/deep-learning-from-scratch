/* ============================================================================
 * Module 05 — ResNet (let the gradient skip; train a network of any depth)
 *
 * The degradation problem and its fix, in miniature. nanograd learned batch
 * normalization, the residual add, the option-(A) shortcut, global average
 * pooling and bias-free layers (lib/c/nanograd/norm.c, nn.c, conv.c); this
 * program trains a 34-layer network on two interleaved spirals TWICE, from
 * the same initial weights:
 *
 *   stem   linear(2->16) -> ReLU
 *   16 x   linear(16->16) -> BN -> ReLU -> linear(16->16) -> BN
 *          -> [+ the block's input]  -> ReLU
 *   head   linear(16->2) -> softmax cross-entropy
 *   full-batch SGD, learning rate 0.05, momentum 0.9, 400 steps.
 *
 * The plain twin skips the bracketed add; the residual twin keeps it. That
 * one ng_add (and its mirror image in the backward pass) is the whole
 * difference -- same parameters, same arithmetic otherwise -- and it decides
 * whether 34 layers can fit 128 points.
 *
 * Every op runs in the same order as the Python mirror (python/resnet.py,
 * class ToyResNet), from the same 64-bit LCG streams, so the C<->Python
 * agreement test pins every printed number (1e-9 gate). Two more FINAL lines
 * check the conv-side machinery on fixed tensors: BatchNorm over NCHW maps
 * (training and evaluation mode), and a downsampling conv residual block with
 * the zero-padding shortcut and global average pooling.
 *
 * Build (standalone, from repo root; Windows: omit -lm, math is in the CRT):
 *   clang -O2 -std=c11 -Wall -Wextra -I lib/c/nanograd \
 *     topics/05-resnet/c/resnet.c \
 *     lib/c/nanograd/rng.c lib/c/nanograd/nn.c lib/c/nanograd/init.c \
 *     lib/c/nanograd/optim.c lib/c/nanograd/conv.c lib/c/nanograd/norm.c \
 *     -o resnet
 * Or via CMake:  cmake -B build && cmake --build build
 *
 * The CIFAR-10 ResNets live in the Python mirror (vectorized, float32,
 * checked by metric); this file is the bit-exact gate.
 *
 * Reference: docs/references/papers.md (Module 05) -- He, Zhang, Ren & Sun
 * (2015), "Deep Residual Learning for Image Recognition"; Ioffe & Szegedy
 * (2015), "Batch Normalization".
 * ==========================================================================*/
#include "nanograd.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define N_PTS     128         /* training points (half per spiral)           */
#define TURNS     1.5         /* how far the spirals wind round each other  */
#define WIDTH     16          /* units per layer                            */
#define DEPTH     16          /* blocks: 2*DEPTH + 2 = 34 weight layers      */
#define N_OUT     2
#define TOY_SEED  1           /* weight draws                               */
#define DATA_SEED 2           /* the spirals                                */
#define TOY_STEPS 400
#define TOY_LR    0.05
#define TOY_MOM   0.9
#define BN_EPS    1e-5
#define BN_MOM    0.1

/* pi as Python's math.pi rounds it (rng.c keeps its own private copy). */
#define PI 3.14159265358979323846

#define ACT (N_PTS * WIDTH)   /* one layer's activations for the whole set  */

/* The spirals, in python/resnet.py make_spirals()'s draw order: point i has
 * class i % 2 and one uniform t -- radius 0.15 + 0.85 t, angle
 * 2*pi*TURNS*t + pi*class (class 1 is class 0 turned half a revolution). */
static void make_spirals(double *X, int *y, int n, uint64_t seed) {
    NgRng r;
    ng_rng_seed(&r, seed);
    for (int i = 0; i < n; i++) {
        y[i] = i % 2;
        const double t = ng_rng_uniform(&r);
        const double rad = 0.15 + 0.85 * t;
        const double a = 2.0 * PI * TURNS * t + PI * (double)y[i];
        X[2 * i] = rad * cos(a);
        X[2 * i + 1] = rad * sin(a);
    }
}

/* One block's parameters (no biases: BatchNorm's beta is the shift) and its
 * BatchNorms' running statistics. */
typedef struct {
    double W1[WIDTH * WIDTH], g1[WIDTH], be1[WIDTH];
    double W2[WIDTH * WIDTH], g2[WIDTH], be2[WIDTH];
} Block;

typedef struct {
    double W0[2 * WIDTH], b0[WIDTH];
    Block  blk[DEPTH];
    double Wh[WIDTH * N_OUT], bh[N_OUT];
} Params;

typedef struct {
    double rm1[WIDTH], rv1[WIDTH], rm2[WIDTH], rv2[WIDTH];
} Running;

/* Everything forward caches for backward, for the whole data set. */
typedef struct {
    double H0[ACT];                     /* stem pre-activation               */
    double A[DEPTH + 1][ACT];           /* block inputs; A[DEPTH] -> head    */
    double U1[DEPTH][ACT], XH1[DEPTH][ACT], IS1[DEPTH][WIDTH];
    double A1[DEPTH][ACT];
    double XH2[DEPTH][ACT], IS2[DEPTH][WIDTH];
    double S[DEPTH][ACT];               /* body (+ input): the ReLU's input  */
    double LOGITS[N_PTS * N_OUT], P[N_PTS * N_OUT];
} Acts;

/* Scratch for one block's forward (the pre-BN values) and for backward. */
typedef struct {
    double Z[ACT], U2[ACT];
    double DA[ACT], DS[ACT], DZ2[ACT], DA1[ACT], DU1[ACT], DZ1[ACT], DAIN[ACT];
    double DH0[ACT], DLOGITS[N_PTS * N_OUT];
} Scratch;

/* Forward pass over the whole set; returns the mean cross-entropy.
 * training != 0: BatchNorm uses (and updates the running averages from) the
 * batch's statistics; 0: it uses the running averages. */
static double forward(const Params *w, Running *run, Acts *a, Scratch *s,
                      const double *X, const int *y, int residual, int training) {
    ng_linear_forward(X, w->W0, w->b0, a->H0, N_PTS, 2, WIDTH);
    ng_relu_forward(a->H0, a->A[0], ACT);
    for (int d = 0; d < DEPTH; d++) {
        const Block *b = &w->blk[d];
        Running *r = &run[d];
        ng_linear_forward(a->A[d], b->W1, NULL, s->Z, N_PTS, WIDTH, WIDTH);
        ng_batchnorm_forward(s->Z, b->g1, b->be1, a->U1[d], a->XH1[d], a->IS1[d],
                             r->rm1, r->rv1, N_PTS, WIDTH, 1, BN_EPS, BN_MOM, training);
        ng_relu_forward(a->U1[d], a->A1[d], ACT);
        ng_linear_forward(a->A1[d], b->W2, NULL, s->Z, N_PTS, WIDTH, WIDTH);
        ng_batchnorm_forward(s->Z, b->g2, b->be2, s->U2, a->XH2[d], a->IS2[d],
                             r->rm2, r->rv2, N_PTS, WIDTH, 1, BN_EPS, BN_MOM, training);
        /* The residual connection: add the block's input back. */
        if (residual)
            ng_add(s->U2, a->A[d], a->S[d], ACT);
        else
            memcpy(a->S[d], s->U2, sizeof a->S[d]);
        ng_relu_forward(a->S[d], a->A[d + 1], ACT);
    }
    ng_linear_forward(a->A[DEPTH], w->Wh, w->bh, a->LOGITS, N_PTS, WIDTH, N_OUT);
    return ng_softmax_ce_forward(a->LOGITS, y, a->P, N_PTS, N_OUT);
}

/* The reversed walk, after a training-mode forward; fills g. */
static void backward(const Params *w, const Acts *a, Scratch *s, Params *g,
                     const double *X, const int *y, int residual) {
    ng_softmax_ce_backward(a->P, y, s->DLOGITS, N_PTS, N_OUT);
    ng_linear_backward(a->A[DEPTH], w->Wh, s->DLOGITS, s->DA, g->Wh, g->bh,
                       N_PTS, WIDTH, N_OUT);
    for (int d = DEPTH - 1; d >= 0; d--) {
        const Block *b = &w->blk[d];
        Block *gb = &g->blk[d];
        ng_relu_backward(a->S[d], s->DA, s->DS, ACT);
        ng_batchnorm_backward(a->XH2[d], b->g2, a->IS2[d], s->DS, s->DZ2,
                              gb->g2, gb->be2, N_PTS, WIDTH, 1, 1);
        ng_linear_backward(a->A1[d], b->W2, s->DZ2, s->DA1, gb->W2, NULL,
                           N_PTS, WIDTH, WIDTH);
        ng_relu_backward(a->U1[d], s->DA1, s->DU1, ACT);
        ng_batchnorm_backward(a->XH1[d], b->g1, a->IS1[d], s->DU1, s->DZ1,
                              gb->g1, gb->be1, N_PTS, WIDTH, 1, 1);
        ng_linear_backward(a->A[d], b->W1, s->DZ1, s->DAIN, gb->W1, NULL,
                           N_PTS, WIDTH, WIDTH);
        /* The fork: the block's input went into the body AND round the
         * shortcut, so its gradient is the sum of both routes back -- the
         * shortcut's share, DS, arrives without passing through any layer. */
        if (residual)
            ng_add(s->DAIN, s->DS, s->DA, ACT);
        else
            memcpy(s->DA, s->DAIN, sizeof s->DA);
    }
    ng_relu_backward(a->H0, s->DA, s->DH0, ACT);
    ng_linear_backward(X, w->W0, s->DH0, NULL, g->W0, g->b0, N_PTS, 2, WIDTH);
}

/* SGD + momentum over every parameter (a Params struct is one flat run of
 * doubles, laid out in the canonical order). */
static void sgd(Params *w, const Params *g, Params *vel) {
    const int n = (int)(sizeof(Params) / sizeof(double));
    ng_sgd_step((double *)w, (const double *)g, (double *)vel, n,
                TOY_LR, TOY_MOM, 0.0);
}

static double accuracy(const Acts *a, const int *y) {
    int correct = 0;
    for (int i = 0; i < N_PTS; i++) {
        const int pred = a->P[i * N_OUT + 1] > a->P[i * N_OUT] ? 1 : 0;
        if (pred == y[i]) correct++;
    }
    return (double)correct / (double)N_PTS;
}

/* Train one twin from TOY_SEED's weights and print its FINAL line. */
static void run_toy(const double *X, const int *y, int residual) {
    static Params w, g, vel;
    static Running run[DEPTH];
    static Acts a;
    static Scratch s;
    memset(&g, 0, sizeof g);
    memset(&vel, 0, sizeof vel);

    /* He init, one stream: W0, then each block's W1 and W2, then Wh.
     * Gammas start at 1, betas and biases at 0, running stats at (0, 1). */
    NgRng rw;
    ng_rng_seed(&rw, TOY_SEED);
    ng_init_linear(w.W0, w.b0, 2, WIDTH, &rw, NG_INIT_HE);
    for (int d = 0; d < DEPTH; d++) {
        Block *b = &w.blk[d];
        ng_init_linear(b->W1, NULL, WIDTH, WIDTH, &rw, NG_INIT_HE);
        ng_init_linear(b->W2, NULL, WIDTH, WIDTH, &rw, NG_INIT_HE);
        for (int j = 0; j < WIDTH; j++) {
            b->g1[j] = b->g2[j] = 1.0;
            b->be1[j] = b->be2[j] = 0.0;
            run[d].rm1[j] = run[d].rm2[j] = 0.0;
            run[d].rv1[j] = run[d].rv2[j] = 1.0;
        }
    }
    ng_init_linear(w.Wh, w.bh, WIDTH, N_OUT, &rw, NG_INIT_HE);

    printf("[toy %s] %d layers (%d blocks of width %d%s); full-batch SGD lr=%g "
           "momentum=%g\n", residual ? "res  " : "plain", 2 * DEPTH + 2, DEPTH,
           WIDTH, residual ? ", each adding its input back" : "", TOY_LR, TOY_MOM);
    double batch_loss = 0.0;
    for (int t = 0; t < TOY_STEPS; t++) {
        batch_loss = forward(&w, run, &a, &s, X, y, residual, 1);
        backward(&w, &a, &s, &g, X, y, residual);
        sgd(&w, &g, &vel);
        if (t % (TOY_STEPS / 4) == 0 || t == TOY_STEPS - 1)
            printf("  step %4d  batch_loss=%.6f\n", t, batch_loss);
    }

    /* Evaluation mode: BatchNorm's running statistics. */
    const double loss = forward(&w, run, &a, &s, X, y, residual, 0);
    const double acc = accuracy(&a, y);

    double wsum = 0.0;                     /* every parameter, canonical order */
    const double *p = (const double *)&w;
    for (int j = 0; j < (int)(sizeof(Params) / sizeof(double)); j++) wsum += p[j];

    printf("FINAL %s loss=%.17g acc=%.17g batch_loss=%.17g wsum=%.17g "
           "w0_00=%.17g w1_00=%.17g g2_last=%.17g rv2_last=%.17g\n",
           residual ? "res" : "plain", loss, acc, batch_loss, wsum,
           w.W0[0], w.blk[0].W1[0], w.blk[DEPTH - 1].g2[0], run[DEPTH - 1].rv2[0]);
}

/* ---- Fixed-tensor checks of the conv-side machinery ---------------------- */

/* n draws of scale*signed + shift, in order (python/resnet.py _draw). */
static void draw(NgRng *r, double *A, int n, double scale, double shift) {
    for (int j = 0; j < n; j++) A[j] = scale * ng_rng_signed(r) + shift;
}

/* Plain and position-weighted sums, printed as name_sum / name_wsum. */
static void sums(const char *name, const double *A, int n) {
    double s = 0.0, ws = 0.0;
    for (int j = 0; j < n; j++) {
        s += A[j];
        ws += A[j] * (double)(j + 1);
    }
    printf(" %s_sum=%.17g %s_wsum=%.17g", name, s, name, ws);
}

/* BatchNorm over (2, 3, 4, 4) NCHW maps: train-mode forward + backward (which
 * updates the running stats), then eval-mode forward + backward on a second
 * input -- python/resnet.py bn_fingerprint(). */
static void bn_check(void) {
    enum { N = 2, C = 3, HW = 16, SZ = N * C * HW };
    double X[SZ], gamma[C], beta[C], dY[SZ], X2[SZ], dY2[SZ];
    double Y[SZ], XH[SZ], IS[C], dX[SZ], dg[C], db[C];
    double Y2[SZ], XH2[SZ], IS2[C], dX2[SZ], dg2[C], db2[C];
    double rm[C] = {0.0, 0.0, 0.0}, rv[C] = {1.0, 1.0, 1.0};
    NgRng r;
    ng_rng_seed(&r, 6);
    draw(&r, X, SZ, 3.0, 1.0);
    draw(&r, gamma, C, 0.5, 1.0);
    draw(&r, beta, C, 1.0, 0.0);
    draw(&r, dY, SZ, 1.0, 0.0);
    draw(&r, X2, SZ, 3.0, 1.0);
    draw(&r, dY2, SZ, 1.0, 0.0);
    ng_batchnorm_forward(X, gamma, beta, Y, XH, IS, rm, rv, N, C, HW, BN_EPS, BN_MOM, 1);
    ng_batchnorm_backward(XH, gamma, IS, dY, dX, dg, db, N, C, HW, 1);
    ng_batchnorm_forward(X2, gamma, beta, Y2, XH2, IS2, rm, rv, N, C, HW, BN_EPS, BN_MOM, 0);
    ng_batchnorm_backward(XH2, gamma, IS2, dY2, dX2, dg2, db2, N, C, HW, 0);
    printf("FINAL bn");
    sums("y", Y, SZ);
    sums("dx", dX, SZ);
    sums("dg", dg, C);
    sums("db", db, C);
    sums("rm", rm, C);
    sums("rv", rv, C);
    sums("ye", Y2, SZ);
    sums("dxe", dX2, SZ);
    printf("\n");
}

/* A downsampling basic block -- conv 3x3/2 (2->4) -> BN -> ReLU -> conv 3x3
 * (4->4) -> BN, plus the zero-padding shortcut, -> ReLU -- then global average
 * pooling, forward and backward on fixed tensors (python/resnet.py
 * block_fingerprint()). */
static void block_check(void) {
    enum { N = 2, CI = 2, H = 6, CO = 4, OH = 3, S = 2,
           NX = N * CI * H * H, NY = N * CO * OH * OH, NK1 = CO * CI * 9, NK2 = CO * CO * 9 };
    double K1[NK1], K2[NK2], g1[CO], b1[CO], g2[CO], b2[CO];
    double X[NX], dOut[N * CO];
    double Z1[NY], U1[NY], XH1[NY], IS1[CO], A1[NY], Z2[NY], U2[NY], XH2[NY], IS2[CO];
    double SC[NY], Sum[NY], A2[NY], Out[N * CO];
    double rm1[CO] = {0}, rv1[CO] = {1, 1, 1, 1}, rm2[CO] = {0}, rv2[CO] = {1, 1, 1, 1};
    double dA2[NY], dS[NY], dZ2[NY], dA1[NY], dU1[NY], dZ1[NY], dXb[NX], dXs[NX], dX[NX];
    double dK1[NK1], dK2[NK2], dg1[CO], db1[CO], dg2[CO], db2[CO];
    NgRng r;
    ng_rng_seed(&r, 7);
    ng_init_conv2d(K1, NULL, CO, CI, 3, 3, &r, NG_INIT_HE);
    ng_init_conv2d(K2, NULL, CO, CO, 3, 3, &r, NG_INIT_HE);
    draw(&r, g1, CO, 0.5, 1.0);
    draw(&r, b1, CO, 1.0, 0.0);
    draw(&r, g2, CO, 0.5, 1.0);
    draw(&r, b2, CO, 1.0, 0.0);
    draw(&r, X, NX, 2.0, 0.0);
    draw(&r, dOut, N * CO, 1.0, 0.0);

    /* Forward: the body, the shortcut, their sum, the ReLU, the pool. */
    ng_conv2d_forward_strided(X, K1, NULL, Z1, N, CI, H, H, CO, 3, 3, S, 1);
    ng_batchnorm_forward(Z1, g1, b1, U1, XH1, IS1, rm1, rv1, N, CO, OH * OH, BN_EPS, BN_MOM, 1);
    ng_relu_forward(U1, A1, NY);
    ng_conv2d_forward_strided(A1, K2, NULL, Z2, N, CO, OH, OH, CO, 3, 3, 1, 1);
    ng_batchnorm_forward(Z2, g2, b2, U2, XH2, IS2, rm2, rv2, N, CO, OH * OH, BN_EPS, BN_MOM, 1);
    ng_shortcut_pad_forward(X, SC, N, CI, H, H, CO, S);
    ng_add(U2, SC, Sum, NY);
    ng_relu_forward(Sum, A2, NY);
    ng_global_avgpool_forward(A2, Out, N, CO, OH, OH);

    /* Backward: the same walk reversed; the input's two gradients add. */
    ng_global_avgpool_backward(dOut, dA2, N, CO, OH, OH);
    ng_relu_backward(Sum, dA2, dS, NY);
    ng_batchnorm_backward(XH2, g2, IS2, dS, dZ2, dg2, db2, N, CO, OH * OH, 1);
    ng_conv2d_backward_strided(A1, K2, dZ2, dA1, dK2, NULL, N, CO, OH, OH, CO, 3, 3, 1, 1);
    ng_relu_backward(U1, dA1, dU1, NY);
    ng_batchnorm_backward(XH1, g1, IS1, dU1, dZ1, dg1, db1, N, CO, OH * OH, 1);
    ng_conv2d_backward_strided(X, K1, dZ1, dXb, dK1, NULL, N, CI, H, H, CO, 3, 3, S, 1);
    ng_shortcut_pad_backward(dS, dXs, N, CI, H, H, CO, S);
    ng_add(dXb, dXs, dX, NX);

    printf("FINAL block");
    sums("y", Out, N * CO);
    sums("dx", dX, NX);
    sums("dk1", dK1, NK1);
    sums("dk2", dK2, NK2);
    sums("dg1", dg1, CO);
    sums("db2", db2, CO);
    printf("\n");
}

int main(void) {
    static double X[N_PTS * 2];
    int y[N_PTS];
    make_spirals(X, y, N_PTS, DATA_SEED);

    run_toy(X, y, 0);      /* the plain network */
    run_toy(X, y, 1);      /* its residual twin, from the same weights */
    bn_check();
    block_check();
    return 0;
}
