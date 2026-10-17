/* ============================================================================
 * Module 04 — AlexNet (depth + ReLU + dropout + augmentation, at scale)
 *
 * The 2012 recipe in miniature. nanograd learned stride and padding for its
 * convolutions, overlapping max-pooling, inverted dropout, local response
 * normalization, and random crop+flip augmentation (lib/c/nanograd/conv.c,
 * nn.c, norm.c, augment.c); this program trains the smallest network that
 * uses AlexNet's whole training recipe:
 *
 *   random 10x10 crop (+ mirror) of a 12x12 RGB image
 *   -> conv(3->6, 5x5, stride 2, pad 2) -> ReLU -> maxpool 3x3 / stride 2
 *   -> linear(24->16) -> ReLU -> dropout(0.5) -> linear(16->2)
 *   SGD with momentum 0.9 and weight decay 5e-4, full batch.
 *
 * The data is "mirror glyphs": a corner shape whose horizontal stroke is at
 * the bottom (class 0) or top (class 1). All 32 training glyphs face one way;
 * all 64 test glyphs are mirrored. Memorizing the training images teaches
 * nothing about the test images -- crops and flips plus dropout do.
 *
 * Every op runs in the same order as the Python mirror (python/alexnet.py,
 * class ToyAlexNet), drawing data, weights, crops and dropout masks from the
 * same 64-bit LCG streams, so the C<->Python agreement test pins every number
 * to a tight relative tolerance (1e-9 gate).
 *
 * A second FINAL line checks the library's LRN (AlexNet's local response
 * normalization -- historical, so the toy doesn't use it) on a fixed tensor.
 *
 * Build (standalone, from repo root; Windows: omit -lm, math is in the CRT):
 *   clang -O2 -std=c11 -Wall -Wextra -I lib/c/nanograd \
 *     topics/04-alexnet/c/alexnet.c \
 *     lib/c/nanograd/rng.c lib/c/nanograd/nn.c lib/c/nanograd/init.c \
 *     lib/c/nanograd/optim.c lib/c/nanograd/conv.c lib/c/nanograd/norm.c \
 *     lib/c/nanograd/augment.c -o alexnet
 * Or via CMake:  cmake -B build && cmake --build build
 *
 * The CIFAR-10 network lives in the Python mirror (vectorized, float32,
 * checked by metric); this file is the bit-exact gate.
 *
 * Reference: docs/references/papers.md (Module 04) -- Krizhevsky, Sutskever
 * & Hinton (2012), "ImageNet Classification with Deep Convolutional Neural
 * Networks".
 * ==========================================================================*/
#include "nanograd.h"

#include <math.h>
#include <stdio.h>

#define N_TRAIN   32
#define N_TEST    64
#define IMG       12          /* image height = width                         */
#define GLYPH     5           /* the corner shape is GLYPH x GLYPH            */
#define CROP      10          /* network input: a 10x10 crop                  */
#define N_CH      3           /* RGB                                          */
#define N_F       6           /* conv filters                                 */
#define KS        5           /* kernel size                                  */
#define STRIDE    2
#define PAD       2
#define CONV_OUT  ((CROP + 2 * PAD - KS) / STRIDE + 1)   /* 5x5 feature maps  */
#define POOL_P    3                                      /* overlapping pool: */
#define POOL_S    2                                      /* 3x3 every 2       */
#define POOL_OUT  ((CONV_OUT - POOL_P) / POOL_S + 1)     /* 2x2 after pooling */
#define N_FLAT    (N_F * POOL_OUT * POOL_OUT)            /* 24 to the head    */
#define N_HID     16
#define N_OUT     2
#define DROP_P    0.5

#define TOY_SEED  1           /* weight draws                                 */
#define DATA_SEED 2           /* training glyphs                              */
#define TEST_SEED 3           /* mirrored test glyphs                         */
#define AUG_SEED  4           /* crop + flip draws                            */
#define DROP_SEED 5           /* dropout masks                                */
#define TOY_STEPS 600
#define SGD_LR    0.02
#define SGD_MOM   0.9
#define SGD_WD    5e-4

#define N_IMG_PX  (N_CH * IMG * IMG)
#define N_CROP_PX (N_CH * CROP * CROP)
#define N_K1      (N_F * N_CH * KS * KS)
#define N_Z       (N_F * CONV_OUT * CONV_OUT)

/* The glyph dataset, in the exact draw order python/alexnet.py's
 * make_glyphs() uses. Per image: top and left corner (each in 1..6, so the
 * evaluation center crop keeps the whole glyph), an RGB tint in [0.5, 1),
 * then N_CH*IMG*IMG uniforms for the noise floor; the glyph's pixels are then
 * overwritten with the tint. Class is i % 2: the horizontal stroke sits at
 * the bottom (0) or top (1); `mirrored` moves the vertical stroke from the
 * left column to the right one, which never changes the class. */
static void make_glyphs(double *X, int *y, int n, uint64_t seed, int mirrored) {
    NgRng rd;
    ng_rng_seed(&rd, seed);
    for (int i = 0; i < n; i++) {
        y[i] = i % 2;
        const int top = 1 + (int)(ng_rng_uniform(&rd) * (double)(IMG - GLYPH - 1));
        const int left = 1 + (int)(ng_rng_uniform(&rd) * (double)(IMG - GLYPH - 1));
        double tint[N_CH];
        for (int c = 0; c < N_CH; c++) tint[c] = 0.5 + 0.5 * ng_rng_uniform(&rd);
        double *img = X + i * N_IMG_PX;
        for (int j = 0; j < N_IMG_PX; j++) img[j] = 0.25 * ng_rng_uniform(&rd);
        const int vcol = mirrored ? GLYPH - 1 : 0;
        const int hrow = y[i] == 0 ? GLYPH - 1 : 0;
        for (int c = 0; c < N_CH; c++)
            for (int r = 0; r < GLYPH; r++)
                for (int k = 0; k < GLYPH; k++)
                    if (k == vcol || r == hrow)
                        img[(c * IMG + (top + r)) * IMG + (left + k)] = tint[c];
    }
}

/* The network's parameters and their SGD velocities. */
typedef struct {
    double K1[N_K1], b1[N_F];
    double W2[N_FLAT * N_HID], b2[N_HID];
    double W3[N_HID * N_OUT], b3[N_OUT];
} Params;

/* Activations for one batch of up to N_TEST images (forward caches). */
typedef struct {
    double Z[N_TEST * N_Z];            /* conv pre-activations             */
    double A[N_TEST * N_Z];            /* after ReLU                       */
    double POOLED[N_TEST * N_FLAT];    /* == the [n,24] head input         */
    int    IDX[N_TEST * N_FLAT];       /* pool winners (offsets into A)    */
    double H_PRE[N_TEST * N_HID];      /* fc1 pre-activations              */
    double H[N_TEST * N_HID];          /* after ReLU                       */
    double MASK[N_TEST * N_HID];       /* dropout mask                     */
    double D[N_TEST * N_HID];          /* after dropout (fc2's input)      */
    double LOGITS[N_TEST * N_OUT];
    double P[N_TEST * N_OUT];
} Acts;

/* Forward pass on n crops XC:[n,3,10,10]; returns the mean cross-entropy.
 * `drop` != NULL means training: draw dropout masks from it. Note there is no
 * flatten: POOLED[n,6,2,2] already IS the row-major [n,24] matrix. */
static double forward(const Params *w, Acts *a, const double *XC, const int *y,
                      int n, NgRng *drop) {
    ng_conv2d_forward_strided(XC, w->K1, w->b1, a->Z, n, N_CH, CROP, CROP,
                              N_F, KS, KS, STRIDE, PAD);
    ng_relu_forward(a->Z, a->A, n * N_Z);
    ng_maxpool2d_forward_strided(a->A, a->POOLED, a->IDX, n, N_F,
                                 CONV_OUT, CONV_OUT, POOL_P, POOL_S);
    ng_linear_forward(a->POOLED, w->W2, w->b2, a->H_PRE, n, N_FLAT, N_HID);
    ng_relu_forward(a->H_PRE, a->H, n * N_HID);
    const double *fc2_in = a->H;
    if (drop) {
        ng_dropout_forward(a->H, a->D, a->MASK, n * N_HID, DROP_P, drop);
        fc2_in = a->D;
    }
    ng_linear_forward(fc2_in, w->W3, w->b3, a->LOGITS, n, N_HID, N_OUT);
    return ng_softmax_ce_forward(a->LOGITS, y, a->P, n, N_OUT);
}

/* Accuracy of the last forward pass. */
static double accuracy(const Acts *a, const int *y, int n) {
    int correct = 0;
    for (int i = 0; i < n; i++) {
        const int pred = a->P[i * N_OUT + 1] > a->P[i * N_OUT] ? 1 : 0;
        if (pred == y[i]) correct++;
    }
    return (double)correct / (double)n;
}

int main(void) {
    /* Data: training glyphs, mirrored test glyphs (independent streams). */
    static double Xtr[N_TRAIN * N_IMG_PX], Xte[N_TEST * N_IMG_PX];
    int ytr[N_TRAIN], yte[N_TEST];
    make_glyphs(Xtr, ytr, N_TRAIN, DATA_SEED, 0);
    make_glyphs(Xte, yte, N_TEST, TEST_SEED, 1);

    /* Parameters: He init, K1 then W2 then W3 from one stream (as ToyAlexNet). */
    static Params w, g, vel;      /* weights, gradients, momentum (zeroed)   */
    NgRng rw;
    ng_rng_seed(&rw, TOY_SEED);
    ng_init_conv2d(w.K1, w.b1, N_F, N_CH, KS, KS, &rw, NG_INIT_HE);
    ng_init_linear(w.W2, w.b2, N_FLAT, N_HID, &rw, NG_INIT_HE);
    ng_init_linear(w.W3, w.b3, N_HID, N_OUT, &rw, NG_INIT_HE);

    /* Augmentation and dropout draw from their own streams, so neither can
     * shift the other's random numbers. */
    NgRng ra, rdrop;
    ng_rng_seed(&ra, AUG_SEED);
    ng_rng_seed(&rdrop, DROP_SEED);

    static Acts a;
    static double XC[N_TEST * N_CROP_PX];                  /* the crops        */
    static double DLOGITS[N_TRAIN * N_OUT], DD[N_TRAIN * N_HID];
    static double DH[N_TRAIN * N_HID], DZ2[N_TRAIN * N_HID];
    static double DPOOL[N_TRAIN * N_FLAT], DA[N_TRAIN * N_Z], DZ[N_TRAIN * N_Z];

    printf("[toy] crop%d -> conv(3->%d, %dx%d, s%d, p%d) -> ReLU -> maxpool %d/%d "
           "-> fc %d->%d -> ReLU -> dropout %.1f -> fc %d->2; SGD+momentum, "
           "crops+flips, mirror glyphs\n",
           CROP, N_F, KS, KS, STRIDE, PAD, POOL_P, POOL_S, N_FLAT, N_HID,
           DROP_P, N_HID);
    for (int t = 0; t < TOY_STEPS; t++) {
        /* A fresh random crop + mirror of every training image, every step. */
        ng_random_crop_flip(Xtr, XC, N_TRAIN, N_CH, IMG, IMG, CROP, 1, &ra);
        double loss = forward(&w, &a, XC, ytr, N_TRAIN, &rdrop);

        /* Backward: the reversed walk. Dropout gates exactly like ReLU, but
         * with a random mask; overlapping max-pool ADDS the gradients a cell
         * won in several windows; the conv backward skips padding taps. */
        ng_softmax_ce_backward(a.P, ytr, DLOGITS, N_TRAIN, N_OUT);
        ng_linear_backward(a.D, w.W3, DLOGITS, DD, g.W3, g.b3,
                           N_TRAIN, N_HID, N_OUT);
        ng_dropout_backward(a.MASK, DD, DH, N_TRAIN * N_HID);
        ng_relu_backward(a.H_PRE, DH, DZ2, N_TRAIN * N_HID);
        ng_linear_backward(a.POOLED, w.W2, DZ2, DPOOL, g.W2, g.b2,
                           N_TRAIN, N_FLAT, N_HID);
        ng_maxpool2d_backward_strided(a.IDX, DPOOL, DA, N_TRAIN, N_F,
                                      CONV_OUT, CONV_OUT, POOL_P, POOL_S);
        ng_relu_backward(a.Z, DA, DZ, N_TRAIN * N_Z);
        ng_conv2d_backward_strided(XC, w.K1, DZ, NULL, g.K1, g.b1,
                                   N_TRAIN, N_CH, CROP, CROP, N_F, KS, KS,
                                   STRIDE, PAD);

        /* SGD with momentum + weight decay, AlexNet's optimizer (Sec. 5). */
        ng_sgd_step(w.K1, g.K1, vel.K1, N_K1, SGD_LR, SGD_MOM, SGD_WD);
        ng_sgd_step(w.b1, g.b1, vel.b1, N_F, SGD_LR, SGD_MOM, SGD_WD);
        ng_sgd_step(w.W2, g.W2, vel.W2, N_FLAT * N_HID, SGD_LR, SGD_MOM, SGD_WD);
        ng_sgd_step(w.b2, g.b2, vel.b2, N_HID, SGD_LR, SGD_MOM, SGD_WD);
        ng_sgd_step(w.W3, g.W3, vel.W3, N_HID * N_OUT, SGD_LR, SGD_MOM, SGD_WD);
        ng_sgd_step(w.b3, g.b3, vel.b3, N_OUT, SGD_LR, SGD_MOM, SGD_WD);

        if (t % (TOY_STEPS / 6) == 0 || t == TOY_STEPS - 1)
            printf("  step %4d  batch_loss=%.6f\n", t, loss);
    }

    /* Evaluation: center crops, dropout off -- on the training glyphs, then
     * on the mirrored test glyphs the network never saw. */
    ng_center_crop(Xtr, XC, N_TRAIN, N_CH, IMG, IMG, CROP);
    const double loss = forward(&w, &a, XC, ytr, N_TRAIN, NULL);
    const double acc = accuracy(&a, ytr, N_TRAIN);
    ng_center_crop(Xte, XC, N_TEST, N_CH, IMG, IMG, CROP);
    forward(&w, &a, XC, yte, N_TEST, NULL);
    const double test_acc = accuracy(&a, yte, N_TEST);

    /* wsum: a checksum over every parameter, in Python's order. */
    double wsum = 0.0;
    for (int i = 0; i < N_K1; i++)           wsum += w.K1[i];
    for (int i = 0; i < N_F; i++)            wsum += w.b1[i];
    for (int i = 0; i < N_FLAT * N_HID; i++) wsum += w.W2[i];
    for (int i = 0; i < N_HID; i++)          wsum += w.b2[i];
    for (int i = 0; i < N_HID * N_OUT; i++)  wsum += w.W3[i];
    for (int i = 0; i < N_OUT; i++)          wsum += w.b3[i];

    printf("FINAL toy loss=%.17g acc=%.17g test_acc=%.17g wsum=%.17g "
           "k1_0=%.17g w2_00=%.17g w3_00=%.17g b3_0=%.17g\n",
           loss, acc, test_acc, wsum, w.K1[0], w.W2[0], w.W3[0], w.b3[0]);

    /* LRN check: X then dY drawn (2*signed, row-major) from one stream, the
     * same as python/alexnet.py's lrn_fingerprint(). */
    enum { LN = 2, LC = 7, LH = 3, LW = 3, LSZ = LN * LC * LH * LW };
    double LX[LSZ], LDY[LSZ], LY[LSZ], LS[LSZ], LDX[LSZ];
    NgRng rl;
    ng_rng_seed(&rl, 6);
    for (int j = 0; j < LSZ; j++) LX[j] = 2.0 * ng_rng_signed(&rl);
    for (int j = 0; j < LSZ; j++) LDY[j] = 2.0 * ng_rng_signed(&rl);
    ng_lrn_forward(LX, LY, LS, LN, LC, LH, LW, 5, 0.1, 0.75, 2.0);
    ng_lrn_backward(LX, LY, LS, LDY, LDX, LN, LC, LH, LW, 5, 0.1, 0.75);
    double y_sum = 0.0, y_wsum = 0.0, dx_sum = 0.0, dx_wsum = 0.0;
    for (int j = 0; j < LSZ; j++) {
        y_sum += LY[j];
        y_wsum += LY[j] * (double)(j + 1);
        dx_sum += LDX[j];
        dx_wsum += LDX[j] * (double)(j + 1);
    }
    printf("FINAL lrn y_sum=%.17g y_wsum=%.17g dx_sum=%.17g dx_wsum=%.17g "
           "y_0=%.17g dx_0=%.17g\n", y_sum, y_wsum, dx_sum, dx_wsum, LY[0], LDX[0]);
    return 0;
}
