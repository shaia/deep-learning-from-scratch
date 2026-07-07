/* ============================================================================
 * Module 00 — The Perceptron (1943 McCulloch-Pitts, 1957 Rosenblatt)
 *
 * Intuition: a perceptron is the simplest thing that can *learn*. It takes an
 * input vector x, forms a weighted sum z = w . x + b, and fires (predicts 1)
 * when z >= 0, otherwise predicts 0. Learning is Rosenblatt's rule: whenever
 * the prediction is wrong, nudge the weights toward the correct answer.
 *
 *     w <- w + lr * (y - yhat) * x
 *     b <- b + lr * (y - yhat)
 *
 * If the two classes can be separated by a straight line, this rule is
 * guaranteed to find such a line (the Perceptron Convergence Theorem). If they
 * cannot -- the famous XOR problem -- a single perceptron is stuck forever.
 * That wall is exactly what motivates the multi-layer network in Module 01.
 *
 * This file is standalone: C11 + libm only. Read it top to bottom.
 * Reference: docs/references/papers.md (Module 00).
 * ==========================================================================*/

#include <stdio.h>
#include <stdint.h>

/* ----------------------------------------------------------------------------
 * Deterministic RNG (a 64-bit linear congruential generator).
 *
 * We roll our own so the Python mirror can reproduce the *exact* same numbers
 * from the same seed -- that is what lets the C<->Python agreement test hold.
 * State is passed explicitly (no function-static state) so it is thread-safe
 * and reads the same as the Python class.
 * The constants are the well-known Knuth/PCG multiplier and increment.
 * --------------------------------------------------------------------------*/
typedef struct { uint64_t state; } Rng;

static void rng_seed(Rng *r, uint64_t seed) { r->state = seed; }

static uint64_t rng_next_u64(Rng *r) {
    r->state = r->state * 6364136223846793005ULL + 1442695040888963407ULL;
    return r->state;
}

/* Uniform double in [0, 1): take the top 53 bits and divide by 2^53. */
static double rng_uniform(Rng *r) {
    return (double)(rng_next_u64(r) >> 11) * (1.0 / 9007199254740992.0);
}

/* Uniform double in [-1, 1). */
static double rng_signed(Rng *r) { return 2.0 * rng_uniform(r) - 1.0; }

/* ----------------------------------------------------------------------------
 * Dataset. Two 2D blobs, one per class, offset far enough to be linearly
 * separable. We use ONLY arithmetic (no sin/log/sqrt) so the C and Python
 * datasets are bit-for-bit identical. Each point = class center + uniform box.
 * --------------------------------------------------------------------------*/
#define N_PER_CLASS 50
#define N_SAMPLES   (2 * N_PER_CLASS)
#define N_FEATURES  2
#define SPREAD      1.5   /* half-width of the box around each class center   */

/* Fill X (row-major, N_SAMPLES x 2) and y (0/1). Class 0 sits at (-2,-2),
 * class 1 at (+2,+2); the ideal separating line is x0 + x1 = 0. */
static void make_blobs(Rng *r, double X[N_SAMPLES][N_FEATURES], int y[N_SAMPLES]) {
    const double center[2][2] = {{-2.0, -2.0}, {2.0, 2.0}};
    int i = 0;
    for (int c = 0; c < 2; c++) {
        for (int k = 0; k < N_PER_CLASS; k++) {
            X[i][0] = center[c][0] + SPREAD * rng_signed(r);
            X[i][1] = center[c][1] + SPREAD * rng_signed(r);
            y[i] = c;
            i++;
        }
    }
}

/* ----------------------------------------------------------------------------
 * The perceptron itself: two weights and a bias.
 * --------------------------------------------------------------------------*/
typedef struct { double w0, w1, b; } Perceptron;

/* Forward pass: pre-activation z = w.x + b, then the step (Heaviside) unit. */
static double preactivation(const Perceptron *p, double x0, double x1) {
    return p->w0 * x0 + p->w1 * x1 + p->b;   /* z = w0*x0 + w1*x1 + b */
}
static int predict(const Perceptron *p, double x0, double x1) {
    return preactivation(p, x0, x1) >= 0.0 ? 1 : 0;   /* yhat */
}

/* Fraction of samples classified correctly. */
static double accuracy(const Perceptron *p,
                       double X[N_SAMPLES][N_FEATURES], int y[N_SAMPLES], int n) {
    int correct = 0;
    for (int i = 0; i < n; i++)
        if (predict(p, X[i][0], X[i][1]) == y[i]) correct++;
    return (double)correct / (double)n;
}

/* One pass of Rosenblatt's learning rule over all n samples, in fixed order.
 * For each sample: err = y - yhat; then move the weights along err * x. */
static void train_epoch(Perceptron *p, double lr,
                        double X[N_SAMPLES][N_FEATURES], int y[N_SAMPLES], int n) {
    for (int i = 0; i < n; i++) {
        int yhat = predict(p, X[i][0], X[i][1]);
        double err = (double)(y[i] - yhat);       /* -1, 0, or +1 */
        p->w0 += lr * err * X[i][0];
        p->w1 += lr * err * X[i][1];
        p->b  += lr * err;
    }
}

/* ----------------------------------------------------------------------------
 * Experiment 1: the blobs (perceptron succeeds, converges to a line).
 * --------------------------------------------------------------------------*/
static void run_blobs(void) {
    Rng r; rng_seed(&r, 42);
    double X[N_SAMPLES][N_FEATURES];
    int y[N_SAMPLES];
    make_blobs(&r, X, y);

    Perceptron p = {0.0, 0.0, 0.0};
    const double lr = 0.1;
    const int epochs = 20;

    printf("[blobs] training a perceptron on two linearly-separable classes\n");
    for (int e = 0; e < epochs; e++) {
        train_epoch(&p, lr, X, y, N_SAMPLES);
        if (e % 5 == 0 || e == epochs - 1)
            printf("  epoch %2d  acc=%.3f  w=(%.3f, %.3f)  b=%.3f\n",
                   e, accuracy(&p, X, y, N_SAMPLES), p.w0, p.w1, p.b);
    }
    /* Machine-readable line for the agreement test (full double precision). */
    printf("FINAL blobs w0=%.17g w1=%.17g b=%.17g acc=%.17g\n",
           p.w0, p.w1, p.b, accuracy(&p, X, y, N_SAMPLES));
}

/* ----------------------------------------------------------------------------
 * Experiment 2: the XOR wall. XOR is not linearly separable, so no line -- and
 * therefore no single perceptron -- can classify all four points. Accuracy
 * gets stuck. This is the whole reason we need hidden layers (Module 01).
 * --------------------------------------------------------------------------*/
static void run_xor(void) {
    double X[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    int    y[4]    = { 0,      1,      1,      0    };   /* XOR truth table */

    Perceptron p = {0.0, 0.0, 0.0};
    const double lr = 0.1;
    const int epochs = 100;

    printf("\n[xor] training the same perceptron on XOR (not separable)\n");
    for (int e = 0; e < epochs; e++) {
        for (int i = 0; i < 4; i++) {
            int yhat = predict(&p, X[i][0], X[i][1]);
            double err = (double)(y[i] - yhat);
            p.w0 += lr * err * X[i][0];
            p.w1 += lr * err * X[i][1];
            p.b  += lr * err;
        }
    }
    int correct = 0;
    for (int i = 0; i < 4; i++)
        if (predict(&p, X[i][0], X[i][1]) == y[i]) correct++;
    double acc = (double)correct / 4.0;
    printf("  after %d epochs acc=%.3f  <- stuck below 1.0: the XOR wall\n",
           epochs, acc);
    printf("FINAL xor w0=%.17g w1=%.17g b=%.17g acc=%.17g\n",
           p.w0, p.w1, p.b, acc);
}

int main(void) {
    run_blobs();
    run_xor();
    return 0;
}
