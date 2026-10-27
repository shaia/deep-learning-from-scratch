/* ============================================================================
 * Module 06 — RNN (share weights across time)
 *
 * A recurrent network reads a sequence one step at a time and carries a hidden
 * state forward -- its memory -- applying the SAME weights at every step:
 *
 *   h<t> = tanh(x<t> Wxh + h<t-1> Whh + bh),     yhat<t> = softmax(h<t> Why + by)
 *
 * nanograd learned the recurrence and backpropagation through time (BPTT) in
 * lib/c/nanograd/rnn.c, and gradient-norm clipping in optim.c. This program
 * trains a 16-unit RNN on "echo" -- a stream of random symbols from a 4-letter
 * alphabet, where the target at every step is the symbol read 4 steps earlier
 * -- TWICE, from the same initial weights on the same batches:
 *
 *   bptt    full backpropagation through time (Werbos 1990)
 *   elman   the gradient stops at the previous state, which is treated as
 *           just another input (Elman 1990's training)
 *
 *   RNN(4 -> 16) -> linear(16 -> 4) -> softmax cross-entropy on every step >= 4;
 *   a fresh batch of 32 sequences x 24 steps per update; Adam (lr 0.01),
 *   gradient norm clipped at 1; 300 updates. Test: 64 sequences of 96 steps.
 *
 * The forward passes are identical; one argument of ng_rnn_backward (truncate
 * = 0 or 1) is the whole difference, and it decides whether the network can
 * learn to remember at all.
 *
 * Every op runs in the same order as the Python mirror (python/rnn.py, class
 * ToyRNN), from the same 64-bit LCG streams, so the C<->Python agreement test
 * pins every printed number (1e-9 gate). Two more FINAL lines check the
 * library on fixed tensors: the RNN forward and backward (initial state, input
 * gradients, truncation), and gradient clipping.
 *
 * Build (standalone, from repo root; Windows: omit -lm, math is in the CRT):
 *   clang -O2 -std=c11 -Wall -Wextra -I lib/c/nanograd \
 *     topics/06-rnn/c/rnn.c \
 *     lib/c/nanograd/rng.c lib/c/nanograd/nn.c lib/c/nanograd/init.c \
 *     lib/c/nanograd/optim.c lib/c/nanograd/rnn.c -o rnn
 * Or via CMake:  cmake -B build && cmake --build build
 *
 * The character-level Shakespeare model lives in the Python mirror
 * (vectorized, float32, checked by metric); this file is the bit-exact gate.
 *
 * Reference: docs/references/papers.md (Module 06) -- Elman (1990), "Finding
 * Structure in Time"; Werbos (1990), "Backpropagation Through Time".
 * ==========================================================================*/
#include "nanograd.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define V         4           /* alphabet size (one-hot inputs, 4-way outputs) */
#define N_H       16          /* hidden units                                  */
#define T_TOY     24          /* steps per training sequence                   */
#define M_TOY     32          /* sequences per batch                           */
#define T_TEST    96          /* the test set: four times longer than training */
#define M_TEST    64
#define DELAY     4           /* echo the symbol from this many steps back     */
#define TOY_SEED  1           /* weight draws                                  */
#define DATA_SEED 2           /* the training batches                          */
#define TEST_SEED 3           /* the fixed test sequences                      */
#define TOY_STEPS 300
#define TOY_LR    0.01
#define CLIP      1.0
#define ADAM_B1   0.9
#define ADAM_B2   0.999
#define ADAM_EPS  1e-8

/* Parameters, in canonical order -- one flat run of doubles, so Adam and the
 * clipping norm can treat the struct as a single array. */
typedef struct {
    double Wxh[V * N_H], Whh[N_H * N_H], bh[N_H];
    double Why[N_H * V], by[V];
} Params;

#define N_PARAMS ((int)(sizeof(Params) / sizeof(double)))

/* Sized for the larger (test) batch; training uses the first T_TOY*M_TOY. */
typedef struct {
    int    S[T_TEST * M_TEST];                    /* symbols, S[t*m + i]          */
    double X[T_TEST * M_TEST * V];                /* one-hot inputs               */
    double H[T_TEST * M_TEST * N_H];              /* hidden states                */
    double LOGITS[T_TEST * M_TEST * V], P[T_TEST * M_TEST * V];
    int    Y[T_TEST * M_TEST];                    /* targets, steps >= DELAY      */
    double DLOGITS[T_TOY * M_TOY * V];
    double DH[T_TOY * M_TOY * N_H], DZ[T_TOY * M_TOY * N_H];
} Work;

/* T x m symbols, uniform over V, drawn sequence by sequence -- each left to
 * right (python/rnn.py draw_symbols) -- then one-hot encoded. */
static void draw_batch(NgRng *r, Work *w, int T, int m) {
    for (int i = 0; i < m; i++)
        for (int t = 0; t < T; t++)
            w->S[t * m + i] = (int)(V * ng_rng_uniform(r));
    memset(w->X, 0, sizeof(double) * (size_t)(T * m * V));
    for (int j = 0; j < T * m; j++) w->X[j * V + w->S[j]] = 1.0;
}

/* The forward pass: the recurrence over all T steps, then the read-out of
 * every step that has a target (t >= DELAY) and its loss. The target of step
 * t is the symbol of step t - DELAY: Y row (t-DELAY)*m + i. */
static double forward(const Params *p, Work *w, int T, int m) {
    ng_rnn_forward(w->X, NULL, p->Wxh, p->Whh, p->bh, w->H, T, m, V, N_H);
    const int rows = (T - DELAY) * m;
    for (int j = 0; j < rows; j++) w->Y[j] = w->S[j];
    ng_linear_forward(w->H + DELAY * m * N_H, p->Why, p->by, w->LOGITS, rows, N_H, V);
    return ng_softmax_ce_forward(w->LOGITS, w->Y, w->P, rows, V);
}

/* Backward, after a training-mode forward: through the read-out into dH (zero
 * for the steps before DELAY, which have no target), then through time. */
static void backward(const Params *p, Work *w, Params *g, int truncate) {
    const int T = T_TOY, m = M_TOY, rows = (T - DELAY) * m;
    ng_softmax_ce_backward(w->P, w->Y, w->DLOGITS, rows, V);
    memset(w->DH, 0, sizeof w->DH);
    ng_linear_backward(w->H + DELAY * m * N_H, p->Why, w->DLOGITS,
                       w->DH + DELAY * m * N_H, g->Why, g->by, rows, N_H, V);
    /* BPTT. truncate = 1 is the elman twin: no gradient into the past. */
    ng_rnn_backward(w->X, NULL, w->H, p->Wxh, p->Whh, w->DH, w->DZ, NULL, NULL,
                    g->Wxh, g->Whh, g->bh, T, m, V, N_H, truncate);
}

static double accuracy(const Work *w, int rows) {
    int correct = 0;
    for (int j = 0; j < rows; j++) {
        int best = 0;                                 /* argmax, first max wins */
        for (int c = 1; c < V; c++)
            if (w->P[j * V + c] > w->P[j * V + best]) best = c;
        if (best == w->Y[j]) correct++;
    }
    return (double)correct / (double)rows;
}

/* Train one twin from TOY_SEED's weights and print its FINAL line. */
static void run_toy(int truncate) {
    static Params p, g, am, av;                       /* params, grads, Adam's moments */
    static Work w;
    memset(&p, 0, sizeof p);
    memset(&am, 0, sizeof am);
    memset(&av, 0, sizeof av);

    /* Xavier init (tanh), one stream: Wxh, Whh, Why. Biases start at 0. */
    NgRng rw;
    ng_rng_seed(&rw, TOY_SEED);
    ng_init_linear(p.Wxh, p.bh, V, N_H, &rw, NG_INIT_XAVIER);
    ng_init_linear(p.Whh, NULL, N_H, N_H, &rw, NG_INIT_XAVIER);
    ng_init_linear(p.Why, p.by, N_H, V, &rw, NG_INIT_XAVIER);

    const char *name = truncate == 1 ? "elman" : "bptt";
    printf("[toy %-5s] echo the symbol from %d steps back; RNN(%d->%d), %s; Adam lr=%g, "
           "clip %g\n", name, DELAY, V, N_H,
           truncate == 1 ? "gradient stops at the previous state (Elman 1990)"
                         : "full backpropagation through time", TOY_LR, CLIP);
    NgRng rd;
    ng_rng_seed(&rd, DATA_SEED);
    double b1_pow = 1.0, b2_pow = 1.0, batch_loss = 0.0, norm = 0.0;
    int n_clip = 0;
    for (int t = 0; t < TOY_STEPS; t++) {
        draw_batch(&rd, &w, T_TOY, M_TOY);            /* a fresh batch every step */
        batch_loss = forward(&p, &w, T_TOY, M_TOY);
        backward(&p, &w, &g, truncate);
        norm = ng_clip_grad_norm((double *)&g, N_PARAMS, CLIP);
        if (norm > CLIP) n_clip++;
        b1_pow *= ADAM_B1;                            /* bias corrections, by product */
        b2_pow *= ADAM_B2;
        ng_adam_step((double *)&p, (const double *)&g, (double *)&am, (double *)&av,
                     N_PARAMS, TOY_LR, ADAM_B1, ADAM_B2, ADAM_EPS, 0.0,
                     1.0 - b1_pow, 1.0 - b2_pow);
        if (t % (TOY_STEPS / 4) == 0 || t == TOY_STEPS - 1)
            printf("  step %4d  batch_loss=%.6f  |grad|=%.4f\n", t, batch_loss, norm);
    }

    /* The test: longer sequences than any it trained on. */
    NgRng rt;
    ng_rng_seed(&rt, TEST_SEED);
    draw_batch(&rt, &w, T_TEST, M_TEST);
    const double loss = forward(&p, &w, T_TEST, M_TEST);
    const double acc = accuracy(&w, (T_TEST - DELAY) * M_TEST);

    double wsum = 0.0;                                /* every parameter, canonical order */
    const double *q = (const double *)&p;
    for (int j = 0; j < N_PARAMS; j++) wsum += q[j];

    printf("FINAL %s loss=%.17g acc=%.17g batch_loss=%.17g gnorm=%.17g n_clip=%d "
           "wsum=%.17g wxh_00=%.17g whh_00=%.17g why_00=%.17g\n",
           name, loss, acc, batch_loss, norm, n_clip, wsum, p.Wxh[0], p.Whh[0], p.Why[0]);
}

/* ---- Fixed-tensor checks of the library ----------------------------------- */

/* n draws of scale*signed + shift, in order (python/rnn.py _draw). */
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

/* The RNN on (T, m, n_in, n_h) = (5, 3, 4, 6) with an initial state: forward,
 * full-BPTT backward, then a backward truncated every 2 steps --
 * python/rnn.py rnn_fingerprint(). */
static void rnn_check(void) {
    enum { T = 5, M = 3, NI = 4, NH = 6, NX = T * M * NI, NHS = T * M * NH };
    double Wxh[NI * NH], Whh[NH * NH], bh[NH], X[NX], h0[M * NH], dH[NHS];
    double H[NHS], dZ[NHS], dX[NX], dh0[M * NH], dWxh[NI * NH], dWhh[NH * NH], dbh[NH];
    NgRng r;
    ng_rng_seed(&r, 6);
    draw(&r, Wxh, NI * NH, 0.6, 0.0);
    draw(&r, Whh, NH * NH, 0.6, 0.0);
    draw(&r, bh, NH, 0.2, 0.0);
    draw(&r, X, NX, 1.5, 0.0);
    draw(&r, h0, M * NH, 0.8, 0.0);
    draw(&r, dH, NHS, 1.0, 0.0);
    ng_rnn_forward(X, h0, Wxh, Whh, bh, H, T, M, NI, NH);
    ng_rnn_backward(X, h0, H, Wxh, Whh, dH, dZ, dX, dh0, dWxh, dWhh, dbh, T, M, NI, NH, 0);
    printf("FINAL rnn");
    sums("h", H, NHS);
    sums("dx", dX, NX);
    sums("dh0", dh0, M * NH);
    sums("dwxh", dWxh, NI * NH);
    sums("dwhh", dWhh, NH * NH);
    sums("dbh", dbh, NH);
    sums("dz", dZ, NHS);
    ng_rnn_backward(X, h0, H, Wxh, Whh, dH, dZ, NULL, NULL, dWxh, dWhh, dbh,
                    T, M, NI, NH, 2);
    sums("dwhh2", dWhh, NH * NH);
    sums("dz2", dZ, NHS);
    printf("\n");
}

/* Clip two arrays (12 and 30 entries, as one vector) to norm 2, then to norm
 * 100 (a no-op) -- python/rnn.py clip_fingerprint(). */
static void clip_check(void) {
    enum { N = 42 };
    double g[N];
    NgRng r;
    ng_rng_seed(&r, 7);
    draw(&r, g, N, 3.0, 0.0);
    const double n1 = ng_clip_grad_norm(g, N, 2.0);
    const double n2 = ng_clip_grad_norm(g, N, 100.0);
    printf("FINAL clip norm=%.17g norm_after=%.17g", n1, n2);
    sums("g", g, N);
    printf("\n");
}

int main(void) {
    run_toy(0);      /* full backpropagation through time */
    run_toy(1);      /* Elman's one-step truncation, from the same weights */
    rnn_check();
    clip_check();
    return 0;
}
