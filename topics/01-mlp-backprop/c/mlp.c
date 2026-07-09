/* ============================================================================
 * Module 01 — MLP + Backpropagation (Rumelhart, Hinton & Williams 1986)
 *
 * Module 00 ended at the XOR wall: no single line splits XOR, so no single
 * perceptron can learn it. The escape is to stack a *hidden layer* in between
 * and train it with backpropagation -- run the input forward, measure the error
 * at the output, then push that error backward through every weight (the chain
 * rule) and nudge each weight against its own gradient.
 *
 *   forward:   z1 = W1 x + b1;  a1 = sigmoid(z1)
 *              z2 = W2 a1 + b2; a2 = sigmoid(z2) = yhat
 *   loss:      L  = -[ y log a2 + (1-y) log(1-a2) ]     (binary cross-entropy)
 *   backward:  dz2 = a2 - y                             (BCE through sigmoid)
 *              dW2 = dz2 a1;    db2 = dz2
 *              dz1 = (W2 dz2) * a1 (1-a1)               (sigmoid' = a(1-a))
 *              dW1 = dz1 x;     db1 = dz1
 *   update:    theta <- theta - lr * dL/dtheta         (gradient descent)
 *
 * This is a 2 -> 4 -> 1 network solving XOR. It is written out with explicit
 * loops in the same operation order as the Python mirror, from the same seeded
 * RNG, so the C<->Python agreement test holds to full double precision.
 *
 * Standalone: C11 + libm only. Read it top to bottom.
 * Reference: docs/references/papers.md (Module 01).
 * ==========================================================================*/

#include <math.h>
#include <stdint.h>
#include <stdio.h>

/* ----------------------------------------------------------------------------
 * Deterministic RNG (64-bit LCG) -- identical to Module 00, so the Python mirror
 * draws the exact same weights from the same seed. State passed explicitly (no
 * function-static state) so it is thread-safe and reads like the Python class.
 * --------------------------------------------------------------------------*/
typedef struct { uint64_t state; } Rng;

static void rng_seed(Rng *r, uint64_t seed) { r->state = seed; }

static uint64_t rng_next_u64(Rng *r) {
    r->state = r->state * 6364136223846793005ULL + 1442695040888963407ULL;
    return r->state;
}

/* Uniform double in [0, 1): top 53 bits over 2^53. */
static double rng_uniform(Rng *r) {
    return (double)(rng_next_u64(r) >> 11) * (1.0 / 9007199254740992.0);
}

/* Uniform double in [-1, 1). */
static double rng_signed(Rng *r) { return 2.0 * rng_uniform(r) - 1.0; }

/* ----------------------------------------------------------------------------
 * Model configuration and the XOR dataset (arithmetic only -> bit-exact data).
 * --------------------------------------------------------------------------*/
#define N_IN       2
#define N_HIDDEN   4
#define N_SAMPLES  4

/* Hyperparameters: chosen so the fixed-seed run drives XOR to 100%. */
#define XOR_SEED    1
#define INIT_SCALE  1.0
#define XOR_LR      0.5
#define XOR_EPOCHS  20000

/* The four XOR points and their labels. */
static const double XOR_X[N_SAMPLES][N_IN] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
static const double XOR_Y[N_SAMPLES]       = { 0,      1,      1,      0    };

/* ----------------------------------------------------------------------------
 * The network: one hidden layer of N_HIDDEN sigmoid units, one sigmoid output.
 * --------------------------------------------------------------------------*/
typedef struct {
    double W1[N_HIDDEN][N_IN];  /* hidden weights, row j = unit j */
    double b1[N_HIDDEN];        /* hidden biases                  */
    double W2[N_HIDDEN];        /* output weights (single output) */
    double b2;                  /* output bias                    */
} MLP;

static double sigmoid(double z) { return 1.0 / (1.0 + exp(-z)); }

/* Initialize weights by drawing from the RNG in a fixed order: W1 row-major,
 * then b1, then W2, then b2 -- the exact order the Python mirror uses. */
static void mlp_init(MLP *m, Rng *r, double scale) {
    for (int j = 0; j < N_HIDDEN; j++)
        for (int k = 0; k < N_IN; k++)
            m->W1[j][k] = scale * rng_signed(r);
    for (int j = 0; j < N_HIDDEN; j++)
        m->b1[j] = scale * rng_signed(r);
    for (int j = 0; j < N_HIDDEN; j++)
        m->W2[j] = scale * rng_signed(r);
    m->b2 = scale * rng_signed(r);
}

/* Forward pass: input -> hidden (sigmoid) -> output (sigmoid). Fills a1[] with
 * the hidden activations and returns a2 = yhat. */
static double forward(const MLP *m, double x0, double x1, double a1[N_HIDDEN]) {
    for (int j = 0; j < N_HIDDEN; j++) {
        double z1 = m->b1[j] + m->W1[j][0] * x0 + m->W1[j][1] * x1;
        a1[j] = sigmoid(z1);                 /* a1_j = sigmoid(z1_j) */
    }
    double z2 = m->b2;
    for (int j = 0; j < N_HIDDEN; j++)
        z2 += m->W2[j] * a1[j];              /* z2 = W2 . a1 + b2 */
    return sigmoid(z2);                      /* yhat = sigmoid(z2) */
}

static int predict(const MLP *m, double x0, double x1) {
    double a1[N_HIDDEN];
    return forward(m, x0, x1, a1) >= 0.5 ? 1 : 0;
}

/* Mean binary cross-entropy over the dataset. */
static double loss(const MLP *m, const double X[N_SAMPLES][N_IN],
                   const double y[N_SAMPLES], int n) {
    double total = 0.0;
    double a1[N_HIDDEN];
    for (int i = 0; i < n; i++) {
        double a2 = forward(m, X[i][0], X[i][1], a1);
        double yi = y[i];
        total += -(yi * log(a2) + (1.0 - yi) * log(1.0 - a2));
    }
    return total / (double)n;
}

static double accuracy(const MLP *m, const double X[N_SAMPLES][N_IN],
                       const double y[N_SAMPLES], int n) {
    int correct = 0;
    for (int i = 0; i < n; i++)
        if (predict(m, X[i][0], X[i][1]) == (int)y[i]) correct++;
    return (double)correct / (double)n;
}

/* ----------------------------------------------------------------------------
 * Backprop: accumulate the gradient of the mean loss over all samples, then a
 * gradient-descent step. The chain rule is written out block by block; no
 * autograd, no solver -- the reader sees every derivative.
 * --------------------------------------------------------------------------*/
static void sgd_step(MLP *m, const double X[N_SAMPLES][N_IN],
                     const double y[N_SAMPLES], int n, double lr) {
    double dW1[N_HIDDEN][N_IN] = {{0}};
    double db1[N_HIDDEN] = {0};
    double dW2[N_HIDDEN] = {0};
    double db2 = 0.0;
    double a1[N_HIDDEN];

    for (int i = 0; i < n; i++) {
        double x0 = X[i][0], x1 = X[i][1], yi = y[i];
        double a2 = forward(m, x0, x1, a1);

        /* Output layer: for BCE through a sigmoid, dL/dz2 collapses to (a2 - y). */
        double dz2 = a2 - yi;
        for (int j = 0; j < N_HIDDEN; j++)
            dW2[j] += dz2 * a1[j];           /* dL/dW2_j = dz2 * a1_j */
        db2 += dz2;                          /* dL/db2   = dz2 */

        /* Hidden layer: push dz2 back through W2, then through sigmoid'. */
        for (int j = 0; j < N_HIDDEN; j++) {
            double da1 = dz2 * m->W2[j];             /* dL/da1_j = W2_j * dz2 */
            double dz1 = da1 * a1[j] * (1.0 - a1[j]); /* dL/dz1_j = da1 * sigmoid' */
            dW1[j][0] += dz1 * x0;                    /* dL/dW1_j0 = dz1_j * x0 */
            dW1[j][1] += dz1 * x1;                    /* dL/dW1_j1 = dz1_j * x1 */
            db1[j] += dz1;                            /* dL/db1_j  = dz1_j */
        }
    }

    /* Mean over the batch (matches the mean in loss()), then descend. */
    for (int j = 0; j < N_HIDDEN; j++) {
        for (int k = 0; k < N_IN; k++) {
            dW1[j][k] /= (double)n;
            m->W1[j][k] -= lr * dW1[j][k];
        }
        db1[j] /= (double)n;
        m->b1[j] -= lr * db1[j];
        dW2[j] /= (double)n;
        m->W2[j] -= lr * dW2[j];
    }
    db2 /= (double)n;
    m->b2 -= lr * db2;
}

/* ----------------------------------------------------------------------------
 * Experiment: train the MLP on XOR. Backprop learns the hidden layer that a
 * single perceptron never could -- the wall from Module 00 finally falls.
 * --------------------------------------------------------------------------*/
static void run_xor(void) {
    Rng r; rng_seed(&r, XOR_SEED);
    MLP m;
    mlp_init(&m, &r, INIT_SCALE);

    printf("[xor] training a 2->%d->1 sigmoid MLP with backprop\n", N_HIDDEN);
    for (int e = 0; e < XOR_EPOCHS; e++) {
        sgd_step(&m, XOR_X, XOR_Y, N_SAMPLES, XOR_LR);
        if (e % (XOR_EPOCHS / 10) == 0 || e == XOR_EPOCHS - 1)
            printf("  epoch %6d  loss=%.6f  acc=%.3f\n",
                   e, loss(&m, XOR_X, XOR_Y, N_SAMPLES),
                   accuracy(&m, XOR_X, XOR_Y, N_SAMPLES));
    }

    /* Machine-readable fingerprint for the agreement test (full precision).
     * wsum is a checksum over every parameter. */
    double wsum = m.b2;
    for (int j = 0; j < N_HIDDEN; j++) {
        wsum += m.b1[j] + m.W2[j];
        for (int k = 0; k < N_IN; k++) wsum += m.W1[j][k];
    }
    printf("FINAL xor loss=%.17g acc=%.17g wsum=%.17g w1_00=%.17g w2_0=%.17g b2=%.17g\n",
           loss(&m, XOR_X, XOR_Y, N_SAMPLES),
           accuracy(&m, XOR_X, XOR_Y, N_SAMPLES),
           wsum, m.W1[0][0], m.W2[0], m.b2);
}

int main(void) {
    run_xor();
    return 0;
}
