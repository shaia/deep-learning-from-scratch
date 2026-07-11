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
 * from `r` in row-major order; sets b (length n_out) to zero.
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
