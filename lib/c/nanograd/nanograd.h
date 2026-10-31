/* ============================================================================
 * nanograd -- the shared, from-scratch neural-net library for this book (C side).
 *
 * Introduced in Module 02 (Making Deep Nets Trainable) and grown module by module, in
 * lockstep with lib/python/nanograd. It is NOT an autograd engine: every
 * backward function is the chain rule written out by hand, localized to one
 * operation (Module 01 derived the whole thing once; here we just package it).
 *
 * Everything operates on flat, row-major double arrays the caller owns -- no
 * hidden allocation, no globals, no state beyond what you pass in -- so the C
 * results match the NumPy mirror and the code stays traceable.
 *
 * Reference: docs/conventions/c-style.md, docs/references/papers.md (Module 02).
 * ==========================================================================*/
#ifndef NANOGRAD_H
#define NANOGRAD_H

#include <stdint.h>

/* ---------------------------------------------------------------------------
 * Deterministic RNG (64-bit LCG) -- identical stream to lib/python/nanograd's
 * Rng, so weight init is bit-exact across C and Python. `normal()` is Box-Muller
 * with a one-value spare cache (also mirrored) for Gaussian init.
 * ------------------------------------------------------------------------- */
typedef struct {
    uint64_t state;
    int      has_spare;   /* Box-Muller caches its paired sample here */
    double   spare;
} NgRng;

void   ng_rng_seed(NgRng *r, uint64_t seed);
double ng_rng_uniform(NgRng *r);   /* [0, 1) */
double ng_rng_signed(NgRng *r);    /* [-1, 1) */
double ng_rng_normal(NgRng *r);    /* N(0, 1) via Box-Muller */

/* ---------------------------------------------------------------------------
 * Weight initialization. Fills W (n_in x n_out, row-major) in place by drawing
 * from `r` in row-major order; sets b (length n_out) to zero (b may be NULL).
 *   XAVIER: std = sqrt(1/fan_in)  (sigmoid/tanh)
 *   HE:     std = sqrt(2/fan_in)  (ReLU)
 *   SMALL:  uniform [-1, 1)       (the Module-01 baseline, for ablation)
 * ------------------------------------------------------------------------- */
typedef enum { NG_INIT_XAVIER, NG_INIT_HE, NG_INIT_SMALL } NgInit;
void ng_init_linear(double *W, double *b, int n_in, int n_out,
                    NgRng *r, NgInit kind);

/* ---------------------------------------------------------------------------
 * Linear (fully-connected) layer, row-major:
 *   X:[m,n_in]  W:[n_in,n_out]  b:[n_out]  ->  Y:[m,n_out]
 * Gradients are SUMMED over the batch (the 1/m is applied at the loss), matching
 * lib/python/nanograd. dX may be NULL for the first layer. Buffers are distinct.
 * b (and so db) may be NULL: a bias-free layer, as before a BatchNorm, whose
 * beta does the bias's job (Module 05).
 * ------------------------------------------------------------------------- */
void ng_linear_forward(const double *X, const double *W, const double *b,
                       double *Y, int m, int n_in, int n_out);
void ng_linear_backward(const double *X, const double *W, const double *dY,
                        double *dX, double *dW, double *db,
                        int m, int n_in, int n_out);

/* ---------------------------------------------------------------------------
 * Activations, elementwise over n entries. Backward multiplies the incoming
 * gradient dA by the local slope; ReLU keys off the pre-activation Z (Z>0),
 * tanh/sigmoid off the cached output A.
 * ------------------------------------------------------------------------- */
void ng_relu_forward(const double *Z, double *A, int n);
void ng_relu_backward(const double *Z, const double *dA, double *dZ, int n);
void ng_tanh_forward(const double *Z, double *A, int n);
void ng_tanh_backward(const double *A, const double *dA, double *dZ, int n);
void ng_sigmoid_forward(const double *Z, double *A, int n);
void ng_sigmoid_backward(const double *A, const double *dA, double *dZ, int n);

/* ---------------------------------------------------------------------------
 * Softmax + cross-entropy, fused (row-wise over k classes).
 *   logits:[m,k], y:[m] class indices; fills P:[m,k] with the softmax probs.
 *   forward returns the mean cross-entropy; backward writes dlogits=(P-Y)/m.
 * ------------------------------------------------------------------------- */
double ng_softmax_ce_forward(const double *logits, const int *y,
                             double *P, int m, int k);
void   ng_softmax_ce_backward(const double *P, const int *y,
                              double *dlogits, int m, int k);

/* ---------------------------------------------------------------------------
 * Convolution + pooling (Module 03; stride/padding from Module 04). Images are
 * flat row-major NCHW, X[((i*C + c)*H + h)*W + w]; kernels are (F,C,KH,KW),
 * K[((f*C + c)*KH + p)*KW + q] (named K -- W already means image width).
 * "conv" is cross-correlation (no kernel flip). Gradients are SUMMED over the
 * batch (the 1/m lives at the loss); dX may be NULL for the first layer;
 * b/db may be NULL for a bias-free conv (Module 05, before a BatchNorm).
 *
 * The _strided forms take a stride S and zero-padding `pad`:
 *   OH = (H + 2*pad - KH)/S + 1,  OW = (W + 2*pad - KW)/S + 1.
 * The plain forms are Module 03's valid, stride-1 case (S=1, pad=0).
 *
 * Loop-nest contract (keeps C<->Python agreement within tolerance): forward
 * and backward both run i,f,u,v outer / c,p,q inner -- see conv.c.
 * ------------------------------------------------------------------------- */
void ng_conv2d_forward_strided(const double *X, const double *K, const double *b,
                               double *Y, int n, int C, int H, int W,
                               int F, int KH, int KW, int S, int pad);
void ng_conv2d_backward_strided(const double *X, const double *K, const double *dY,
                                double *dX, double *dK, double *db,
                                int n, int C, int H, int W, int F, int KH, int KW,
                                int S, int pad);
void ng_conv2d_forward(const double *X, const double *K, const double *b,
                       double *Y, int n, int C, int H, int W,
                       int F, int KH, int KW);
void ng_conv2d_backward(const double *X, const double *K, const double *dY,
                        double *dX, double *dK, double *db,
                        int n, int C, int H, int W, int F, int KH, int KW);

/* PxP max pooling with window stride S: OH = (H-P)/S + 1. S < P overlaps the
 * windows (AlexNet: P=3, S=2); the plain forms are Module 03's S = P tiles.
 * Max pooling caches each window winner's flat offset into X in `idx` (one per
 * output cell), so backward is a pure gradient scatter-ADD (a cell that wins
 * overlapping windows collects every gradient it won); ties go to the first
 * maximum in p,q scan order, matching NumPy argmax. Average pooling is
 * non-overlapping (H, W divisible by P) and spreads dY/(P*P) evenly. */
void ng_maxpool2d_forward_strided(const double *X, double *Y, int *idx,
                                  int n, int C, int H, int W, int P, int S);
void ng_maxpool2d_backward_strided(const int *idx, const double *dY, double *dX,
                                   int n, int C, int H, int W, int P, int S);
void ng_maxpool2d_forward(const double *X, double *Y, int *idx,
                          int n, int C, int H, int W, int P);
void ng_maxpool2d_backward(const int *idx, const double *dY, double *dX,
                           int n, int C, int H, int W, int P);
void ng_avgpool2d_forward(const double *X, double *Y,
                          int n, int C, int H, int W, int P);
void ng_avgpool2d_backward(const double *dY, double *dX,
                           int n, int C, int H, int W, int P);

/* Kernel init: He/Xavier with conv fan-in = C*KH*KW (SMALL kept for ablation).
 * Fills K flat in row-major (F,C,KH,KW) order from `r`; zeroes b (length F;
 * may be NULL). */
void ng_init_conv2d(double *K, double *b, int F, int C, int KH, int KW,
                    NgRng *r, NgInit kind);

/* ---------------------------------------------------------------------------
 * Regularization + normalization (Module 04, AlexNet).
 *
 * Dropout (inverted), over n entries: draws one uniform per entry from `r` in
 * order; mask[j] = (u < 1-p) ? 1/(1-p) : 0, Y = X*mask. Backward: dX = dY*mask.
 * p == 0 is the identity and draws nothing (as in Python). Evaluation simply
 * skips the call -- the 1/(1-p) scaling already happened.
 *
 * Local response normalization across channels (AlexNet Sec. 3.3), NCHW:
 *   S = k + alpha * sum_{|j-c| <= size/2} X[j]^2,   Y = X * S^-beta
 * S (same shape as X) is cached for backward.
 * ------------------------------------------------------------------------- */
void ng_dropout_forward(const double *X, double *Y, double *mask, int n,
                        double p, NgRng *r);
void ng_dropout_backward(const double *mask, const double *dY, double *dX,
                         int n);
void ng_lrn_forward(const double *X, double *Y, double *S,
                    int n, int C, int H, int W,
                    int size, double alpha, double beta, double k);
void ng_lrn_backward(const double *X, const double *Y, const double *S,
                     const double *dY, double *dX,
                     int n, int C, int H, int W,
                     int size, double alpha, double beta);

/* ---------------------------------------------------------------------------
 * Data augmentation (Module 04). X:[n,C,H,W] -> out:[n,C,crop,crop].
 * Random crop + left-right flip draws three uniforms per image (row offset,
 * column offset, flip) in that order -- the stream augment.py draws -- and
 * mirrors only if `flip` is nonzero. Center crop is the evaluation view.
 * ------------------------------------------------------------------------- */
void ng_random_crop_flip(const double *X, double *out,
                         int n, int C, int H, int W, int crop, int flip,
                         NgRng *r);
void ng_center_crop(const double *X, double *out,
                    int n, int C, int H, int W, int crop);

/* ---------------------------------------------------------------------------
 * Residual networks (Module 05, ResNet).
 *
 * Batch normalization, per channel c over its m = n*HW values, with layout
 * X[(i*C + c)*HW + k] -- HW = 1 after a Linear, H*W after a conv:
 *   xhat = (x - mean) * invstd,  invstd = 1/sqrt(var + eps),  y = gamma*xhat + beta.
 * training != 0 uses the batch's mean and (biased) variance and updates the
 * running averages, run = (1-momentum)*run + momentum*batch (variance
 * unbiased, *m/(m-1)); training == 0 normalizes with the running statistics
 * and leaves them alone. XHAT ([n,C,HW]) and invstd ([C]) are cached for
 * backward, which writes dgamma = sum dY*xhat, dbeta = sum dY and
 *   training:   dX = (gamma*invstd/m) * (m*dY - dbeta - xhat*dgamma)
 *   evaluation: dX = dY * gamma * invstd          (dX may be NULL).
 *
 * ng_add: Y = A + B -- the residual connection forward (body + shortcut) and,
 * at the forked input, its backward (the two gradients summed).
 *
 * Option-(A) shortcut for a block that downsamples by S and widens C -> F:
 * every S-th pixel, OH = (H-1)/S + 1, input channels at offset (F-C)/2, zero
 * planes elsewhere. Global average pooling: Y[i,c] = mean over H*W.
 * ------------------------------------------------------------------------- */
void ng_batchnorm_forward(const double *X, const double *gamma, const double *beta,
                          double *Y, double *XHAT, double *invstd,
                          double *run_mean, double *run_var,
                          int n, int C, int HW, double eps, double momentum,
                          int training);
void ng_batchnorm_backward(const double *XHAT, const double *gamma,
                           const double *invstd, const double *dY,
                           double *dX, double *dgamma, double *dbeta,
                           int n, int C, int HW, int training);
void ng_add(const double *A, const double *B, double *Y, int n);
void ng_shortcut_pad_forward(const double *X, double *Y,
                             int n, int C, int H, int W, int F, int S);
void ng_shortcut_pad_backward(const double *dY, double *dX,
                              int n, int C, int H, int W, int F, int S);
void ng_global_avgpool_forward(const double *X, double *Y,
                               int n, int C, int H, int W);
void ng_global_avgpool_backward(const double *dY, double *dX,
                                int n, int C, int H, int W);

/* ---------------------------------------------------------------------------
 * Recurrent networks (Module 06, RNN).
 *
 * A vanilla tanh RNN over a whole sequence, time-major:
 *   X:[T,m,n_in]  h0:[m,n_h]  Wxh:[n_in,n_h]  Whh:[n_h,n_h]  bh:[n_h]  ->  H:[T,m,n_h]
 *   h<t> = tanh(x<t> Wxh + h<t-1> Whh + bh)       (h0 NULL: the zero state)
 * Backward is backpropagation through time: given dH = dL/dh<t> for every step
 * it writes dZ = dL/dz<t> ([T,m,n_h], the gradient through time), dX and dh0
 * (either may be NULL), and the shared weights' gradients, SUMMED over every
 * step and row. truncate = k > 0 cuts the gradient's path into the past every
 * k steps (k = 1: Elman's 1990 training); 0 is full BPTT. Loop orders are
 * documented in rnn.c -- the Python mirror depends on them.
 *
 * Gradient-norm clipping (Pascanu, Mikolov & Bengio 2013) over a flat array of
 * n gradients: if ||g|| > max_norm, g *= max_norm/||g||. Returns ||g|| before
 * clipping.
 * ------------------------------------------------------------------------- */
void ng_rnn_forward(const double *X, const double *h0,
                    const double *Wxh, const double *Whh, const double *bh,
                    double *H, int T, int m, int n_in, int n_h);
void ng_rnn_backward(const double *X, const double *h0, const double *H,
                     const double *Wxh, const double *Whh, const double *dH,
                     double *dZ, double *dX, double *dh0,
                     double *dWxh, double *dWhh, double *dbh,
                     int T, int m, int n_in, int n_h, int truncate);
double ng_clip_grad_norm(double *g, int n, double max_norm);

/* ---------------------------------------------------------------------------
 * Long short-term memory (Module 07, LSTM).
 *
 * An LSTM layer with forget gate over a whole sequence, time-major:
 *   X:[T,m,n_in]  h0,c0:[m,n_h]  Wx:[n_in,4n_h]  Wh:[n_h,4n_h]  b:[4n_h]
 *     -> G:[T,m,4n_h] (activated gates)  C:[T,m,n_h] (cells)  H:[T,m,n_h]
 *   z = x<t> Wx + h<t-1> Wh + b, blocks [i | f | g | o];  i,f,o = sigmoid, g = tanh
 *   c<t> = f*c<t-1> + i*g,   h<t> = o*tanh(c<t>)     (h0/c0 NULL: zeros)
 * Backward is backpropagation through time along both paths: given dH it
 * writes dZ (the gate gradients, [T,m,4n_h]) and dC = dL/dc<t> ([T,m,n_h], the
 * gradient along the carousel), dX, dh0 and dc0 (any may be NULL), and the
 * shared weights' gradients, SUMMED over every step and row. truncate = k > 0
 * cuts both paths every k steps (chunked training); cell_only != 0 cuts the h
 * path at every step, so the error flows back only along the cell (Hochreiter
 * & Schmidhuber's 1997 truncated gradient). Loop orders are documented in
 * lstm.c -- the Python mirror depends on them.
 * ------------------------------------------------------------------------- */
void ng_lstm_forward(const double *X, const double *h0, const double *c0,
                     const double *Wx, const double *Wh, const double *b,
                     double *G, double *C, double *H, int T, int m, int n_in, int n_h);
void ng_lstm_backward(const double *X, const double *h0, const double *c0,
                      const double *G, const double *C, const double *H,
                      const double *Wx, const double *Wh, const double *dH,
                      double *dZ, double *dC, double *dX, double *dh0, double *dc0,
                      double *dWx, double *dWh, double *db,
                      int T, int m, int n_in, int n_h, int truncate, int cell_only);

/* ---------------------------------------------------------------------------
 * Optimizers, over a flat parameter array of length n. State buffers (velocity
 * / running averages / moments) are caller-owned and zero-initialized once.
 * L2 weight decay adds weight_decay*theta to the gradient.
 *
 * Adam takes the bias-correction denominators b1c=1-beta1^t, b2c=1-beta2^t from
 * the caller, who advances them by multiplication (b_pow*=beta each step) so C
 * and Python never diverge on a pow() ulp.
 * ------------------------------------------------------------------------- */
void ng_sgd_step(double *p, const double *g, double *vel, int n,
                 double lr, double momentum, double weight_decay);
void ng_rmsprop_step(double *p, const double *g, double *s, int n,
                     double lr, double beta, double eps, double weight_decay);
void ng_adam_step(double *p, const double *g, double *m, double *v, int n,
                  double lr, double beta1, double beta2, double eps,
                  double weight_decay, double b1c, double b2c);

#endif /* NANOGRAD_H */
