/* nanograd convolution + pooling -- Module 03's additions to the library, with
 * the stride and padding dials Module 04 (AlexNet) turns.
 *
 * A conv layer is a Linear layer with locality and weight sharing bolted on;
 * these functions are its forward and hand-written backward, plus the two
 * poolings (max for the modern nets, average for LeNet-5's original design).
 *
 * Layout (shared with lib/python/nanograd/conv.py):
 *   images  X[((i*C + c)*H + h)*W + w]     -- flat row-major NCHW
 *   kernels K[((f*C + c)*KH + p)*KW + q]   -- (F, C, KH, KW); named K because
 *                                             W already means image width
 * "conv" here is cross-correlation (no kernel flip) -- the choice every modern
 * framework makes. Gradients are SUMMED over the batch; the 1/m is applied
 * once, at the loss.
 *
 * Stride S and zero-padding `pad`: output cell (u, v) reads the input patch
 * whose top-left corner is (u*S - pad, v*S - pad); taps that land in the
 * padding read zero, so they are simply skipped (adding K*0 changes nothing).
 * OH = (H + 2*pad - KH)/S + 1. Module 03's ng_conv2d_forward/backward are the
 * S=1, pad=0 special case and call straight through -- with no tap ever out
 * of bounds, the arithmetic (and so Module 03's agreement test) is unchanged.
 *
 * Loop-nest contract: i,f,u,v outer / c,p,q inner, in both forward and
 * backward, and the Python scalar mirrors (ToyCNN in topics/03-lenet,
 * ToyAlexNet in topics/04-alexnet) accumulate in exactly this order -- that is
 * what keeps the C<->Python agreement inside its 1e-9 gate (~1e-15 observed)
 * even after hundreds of training steps. */
#include "nanograd.h"

#include <math.h>

/* Y[i,f,u,v] = b[f] + sum_{c,p,q} K[f,c,p,q] * X[i,c,u*S+p-pad,v*S+q-pad]
 * X:[n,C,H,W]  K:[F,C,KH,KW]  b:[F]  ->  Y:[n,F,OH,OW] */
void ng_conv2d_forward_strided(const double *X, const double *K, const double *b,
                               double *Y, int n, int C, int H, int W,
                               int F, int KH, int KW, int S, int pad) {
    /* A kernel larger than the padded input has no valid position. (C's
     * truncating division would otherwise round the negative numerator
     * toward zero and claim one.) */
    if (H + 2 * pad < KH || W + 2 * pad < KW) return;
    const int OH = (H + 2 * pad - KH) / S + 1, OW = (W + 2 * pad - KW) / S + 1;
    for (int i = 0; i < n; i++)
        for (int f = 0; f < F; f++)
            for (int u = 0; u < OH; u++)
                for (int v = 0; v < OW; v++) {
                    double acc = b[f];
                    for (int c = 0; c < C; c++)
                        for (int p = 0; p < KH; p++)
                            for (int q = 0; q < KW; q++) {
                                const int h = u * S + p - pad, w = v * S + q - pad;
                                if (h < 0 || h >= H || w < 0 || w >= W)
                                    continue;            /* a padding zero */
                                acc += K[((f * C + c) * KH + p) * KW + q]
                                     * X[((i * C + c) * H + h) * W + w];
                            }
                    Y[((i * F + f) * OH + u) * OW + v] = acc;
                }
}

/* The three conv gradients, straight from the chain rule:
 *   db[f]        = sum_{i,u,v} dY[i,f,u,v]
 *   dK[f,c,p,q]  = sum_{i,u,v} dY[i,f,u,v] * X[i,c,u*S+p-pad,v*S+q-pad]
 *                  (the input correlated with the upstream gradient)
 *   dX[i,c,h,w]  = sum over the outputs each pixel fed, times the kernel tap
 *                  that carried it (at S=1, pad=0: a full correlation with the
 *                  FLIPPED kernel -- the true convolution reappears here on
 *                  its own). Padding taps fed nothing, so they get nothing.
 * Same i,f,u,v nest as forward; dX may be NULL for the first layer. */
void ng_conv2d_backward_strided(const double *X, const double *K, const double *dY,
                                double *dX, double *dK, double *db,
                                int n, int C, int H, int W, int F, int KH, int KW,
                                int S, int pad) {
    for (int j = 0; j < F * C * KH * KW; j++) dK[j] = 0.0;
    for (int f = 0; f < F; f++) db[f] = 0.0;
    if (dX)
        for (int j = 0; j < n * C * H * W; j++) dX[j] = 0.0;
    /* No valid kernel position (see forward): every gradient is zero. */
    if (H + 2 * pad < KH || W + 2 * pad < KW) return;
    const int OH = (H + 2 * pad - KH) / S + 1, OW = (W + 2 * pad - KW) / S + 1;

    for (int i = 0; i < n; i++)
        for (int f = 0; f < F; f++)
            for (int u = 0; u < OH; u++)
                for (int v = 0; v < OW; v++) {
                    double g = dY[((i * F + f) * OH + u) * OW + v];
                    if (g == 0.0) continue;   /* ReLU/pool leave most cells dead */
                    db[f] += g;
                    for (int c = 0; c < C; c++)
                        for (int p = 0; p < KH; p++)
                            for (int q = 0; q < KW; q++) {
                                const int h = u * S + p - pad, w = v * S + q - pad;
                                if (h < 0 || h >= H || w < 0 || w >= W)
                                    continue;
                                dK[((f * C + c) * KH + p) * KW + q] +=
                                    g * X[((i * C + c) * H + h) * W + w];
                                if (dX)
                                    dX[((i * C + c) * H + h) * W + w] +=
                                        g * K[((f * C + c) * KH + p) * KW + q];
                            }
                }
}

/* Module 03's valid, stride-1 convolution: the special case above. */
void ng_conv2d_forward(const double *X, const double *K, const double *b,
                       double *Y, int n, int C, int H, int W,
                       int F, int KH, int KW) {
    ng_conv2d_forward_strided(X, K, b, Y, n, C, H, W, F, KH, KW, 1, 0);
}

void ng_conv2d_backward(const double *X, const double *K, const double *dY,
                        double *dX, double *dK, double *db,
                        int n, int C, int H, int W, int F, int KH, int KW) {
    ng_conv2d_backward_strided(X, K, dY, dX, dK, db, n, C, H, W, F, KH, KW, 1, 0);
}

/* PxP max pooling with window stride S; OH = (H - P)/S + 1. S = P gives
 * Module 03's non-overlapping tiles; AlexNet's overlapping pooling is P=3,
 * S=2. Keeps only the strongest response per window; `idx` caches the
 * winner's flat offset into X so backward is pure routing. Scan order within
 * a window is p,q ascending with strict >, so ties go to the FIRST maximum --
 * the same tie-break as NumPy argmax in the Python mirror. */
void ng_maxpool2d_forward_strided(const double *X, double *Y, int *idx,
                                  int n, int C, int H, int W, int P, int S) {
    if (H < P || W < P) return;   /* no window fits (see the conv guard) */
    const int OH = (H - P) / S + 1, OW = (W - P) / S + 1;
    for (int i = 0; i < n; i++)
        for (int c = 0; c < C; c++)
            for (int u = 0; u < OH; u++)
                for (int v = 0; v < OW; v++) {
                    double best = 0.0;
                    int best_i = -1;
                    for (int p = 0; p < P; p++)
                        for (int q = 0; q < P; q++) {
                            int xi = ((i * C + c) * H + (u * S + p)) * W
                                     + (v * S + q);
                            if (best_i < 0 || X[xi] > best) {
                                best = X[xi];
                                best_i = xi;
                            }
                        }
                    int o = ((i * C + c) * OH + u) * OW + v;
                    Y[o] = best;
                    idx[o] = best_i;
                }
}

/* Winner takes the whole gradient; every other cell gets zero (the same mask
 * idea as ReLU's backward). dX is zeroed here, then scattered into with +=,
 * so a cell that won two OVERLAPPING windows receives both gradients -- the
 * chain rule's sum over every use of a value. Windows are visited in output
 * order o, which the Python scalar mirror reproduces. */
void ng_maxpool2d_backward_strided(const int *idx, const double *dY, double *dX,
                                   int n, int C, int H, int W, int P, int S) {
    for (int j = 0; j < n * C * H * W; j++) dX[j] = 0.0;
    if (H < P || W < P) return;   /* no window fits: nothing to route */
    const int OH = (H - P) / S + 1, OW = (W - P) / S + 1;
    for (int o = 0; o < n * C * OH * OW; o++)
        dX[idx[o]] += dY[o];
}

/* Module 03's non-overlapping pooling: the S = P special case. */
void ng_maxpool2d_forward(const double *X, double *Y, int *idx,
                          int n, int C, int H, int W, int P) {
    ng_maxpool2d_forward_strided(X, Y, idx, n, C, H, W, P, P);
}

void ng_maxpool2d_backward(const int *idx, const double *dY, double *dX,
                           int n, int C, int H, int W, int P) {
    ng_maxpool2d_backward_strided(idx, dY, dX, n, C, H, W, P, P);
}

/* Non-overlapping PxP average pooling (LeNet-5's original choice, minus its
 * trainable coefficient). Backward spreads dY/(P*P) evenly over each window;
 * windows don't overlap, so each input cell is written exactly once. */
void ng_avgpool2d_forward(const double *X, double *Y,
                          int n, int C, int H, int W, int P) {
    const int OH = H / P, OW = W / P;
    const double inv = 1.0 / (double)(P * P);
    for (int i = 0; i < n; i++)
        for (int c = 0; c < C; c++)
            for (int u = 0; u < OH; u++)
                for (int v = 0; v < OW; v++) {
                    double s = 0.0;
                    for (int p = 0; p < P; p++)
                        for (int q = 0; q < P; q++)
                            s += X[((i * C + c) * H + (u * P + p)) * W
                                   + (v * P + q)];
                    Y[((i * C + c) * OH + u) * OW + v] = s * inv;
                }
}

void ng_avgpool2d_backward(const double *dY, double *dX,
                           int n, int C, int H, int W, int P) {
    const int OH = H / P, OW = W / P;
    const double inv = 1.0 / (double)(P * P);
    for (int i = 0; i < n; i++)
        for (int c = 0; c < C; c++)
            for (int u = 0; u < OH; u++)
                for (int v = 0; v < OW; v++) {
                    double share = dY[((i * C + c) * OH + u) * OW + v] * inv;
                    for (int p = 0; p < P; p++)
                        for (int q = 0; q < P; q++)
                            dX[((i * C + c) * H + (u * P + p)) * W
                               + (v * P + q)] = share;
                }
}

/* Kernel initialization. Same schemes as ng_init_linear but with the conv
 * fan-in: each output unit reads C*KH*KW inputs (every channel of one
 * kernel-sized patch). Fills K in flat row-major (F,C,KH,KW) order -- the
 * exact draw order of lib/python/nanograd/init.py's *_conv2d fills -- and
 * zeroes b (length F). */
void ng_init_conv2d(double *K, double *b, int F, int C, int KH, int KW,
                    NgRng *r, NgInit kind) {
    const int fan_in = C * KH * KW;
    const int total = F * fan_in;
    if (kind == NG_INIT_SMALL) {
        for (int j = 0; j < total; j++) K[j] = ng_rng_signed(r);
    } else {
        double std = (kind == NG_INIT_HE) ? sqrt(2.0 / (double)fan_in)
                                          : sqrt(1.0 / (double)fan_in);
        for (int j = 0; j < total; j++) K[j] = std * ng_rng_normal(r);
    }
    for (int f = 0; f < F; f++) b[f] = 0.0;
}
