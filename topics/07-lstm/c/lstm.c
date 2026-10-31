/* ============================================================================
 * Module 07 — LSTM (gates that protect the gradient)
 *
 * Module 06's recurrent network rewrites its whole state at every step, so the
 * gradient through time is a power of one matrix and fades. Hochreiter and
 * Schmidhuber's long short-term memory (1997) keeps a second state, the cell,
 * that is ADDED to under gates instead of rewritten:
 *
 *   i, f, o = sigmoid(.)   g = tanh(.)    of  z = x<t> Wx + h<t-1> Wh + b
 *   c<t> = f * c<t-1> + i * g               h<t> = o * tanh(c<t>)
 *
 * so one step back along the cell multiplies the gradient by f -- a number
 * the network sets -- and by nothing else. nanograd learned the layer in
 * lib/c/nanograd/lstm.c. This program trains four networks on "the latch":
 * at step 0 a signal (A B C D), then 79 distractors (w x y z), and at the
 * last step the question: which signal came first? The only error signal is
 * at the end, so the lesson "keep that symbol" has to travel 79 steps back.
 *
 *   rnn       Module 06's RNN, 16 units, full backpropagation through time
 *   lstm      an LSTM, 16 cells, forget bias +5, input bias -5: cells that
 *             start out open to the past and shut to new input
 *   carousel  the SAME LSTM weights, trained with the 1997 gradient: the h
 *             path cut at every step, so the error flows back only along
 *             the cell -- the "constant error carousel"
 *   forget1   the SAME Wx, Wh, Why with the textbook biases (forget 1,
 *             input 0)
 *
 *   recurrent layer -> linear(16 -> 4) on the last state -> softmax CE;
 *   a fresh batch of 32 sequences x 80 steps per update; Adam (lr 0.01),
 *   gradient norm clipped at 1; 300 updates. Test: 64 fresh sequences of 80
 *   steps, and 64 of 320.
 *
 * Every op runs in the same order as the Python mirror (python/lstm.py, class
 * ToyLatch), from the same 64-bit LCG streams, so the C<->Python agreement
 * test pins every printed number (1e-9 gate). One more FINAL line checks the
 * library on fixed tensors: the LSTM forward and backward (initial state and
 * cell, input gradients, truncation, the cell-only gradient).
 *
 * Build (standalone, from repo root; Windows: omit -lm, math is in the CRT):
 *   clang -O2 -std=c11 -Wall -Wextra -I lib/c/nanograd \
 *     topics/07-lstm/c/lstm.c \
 *     lib/c/nanograd/rng.c lib/c/nanograd/nn.c lib/c/nanograd/init.c \
 *     lib/c/nanograd/optim.c lib/c/nanograd/rnn.c lib/c/nanograd/lstm.c -o lstm
 * Or via CMake:  cmake -B build && cmake --build build
 *
 * The character-level Shakespeare model lives in the Python mirror
 * (vectorized, float32, checked by metric); this file is the bit-exact gate.
 *
 * Reference: docs/references/papers.md (Module 07) -- Hochreiter &
 * Schmidhuber (1997), "Long Short-Term Memory"; Gers, Schmidhuber & Cummins
 * (2000), "Learning to Forget".
 * ==========================================================================*/
#include "nanograd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NS        4           /* signal symbols A B C D (and the 4-way output)  */
#define ND        4           /* distractor symbols w x y z                     */
#define V         (NS + ND)   /* one-hot input width: signals, then distractors */
#define N_H       16          /* hidden units / cells                           */
#define N4        (4 * N_H)   /* the LSTM's four gate blocks, side by side      */
#define T_TOY     80          /* steps per training sequence                    */
#define M_TOY     32          /* sequences per batch                            */
#define T_TEST    320         /* the long test: four times the training length  */
#define M_TEST    64
#define TOY_SEED  1           /* weight draws                                   */
#define DATA_SEED 2           /* the training batches                           */
#define TEST_SEED 3           /* the fixed test sequences                       */
#define TOY_STEPS 300
#define TOY_LR    0.01
#define CLIP      1.0
#define ADAM_B1   0.9
#define ADAM_B2   0.999
#define ADAM_EPS  1e-8

typedef enum { TWIN_RNN, TWIN_LSTM, TWIN_CAROUSEL, TWIN_FORGET1 } Twin;
static const char *const TWIN_NAME[] = {"rnn", "lstm", "carousel", "forget1"};

/* Parameters, in canonical order -- one flat run of doubles each, so Adam and
 * the clipping norm can treat a struct as a single array. */
typedef struct {
    double Wxh[V * N_H], Whh[N_H * N_H], bh[N_H];
    double Why[N_H * NS], by[NS];
} RnnParams;
typedef struct {
    double Wx[V * N4], Wh[N_H * N4], b[N4];
    double Why[N_H * NS], by[NS];
} LstmParams;

/* Sized for the long test; training uses the first T_TOY*M_TOY rows. About
 * 18 MB, so main() allocates it once and lends it to each run. */
typedef struct {
    int    S[T_TEST * M_TEST];                    /* symbols, S[t*m + i]            */
    double X[T_TEST * M_TEST * V];                /* one-hot inputs                 */
    double G[T_TEST * M_TEST * N4];               /* the LSTM's activated gates     */
    double C[T_TEST * M_TEST * N_H];              /* the LSTM's cells               */
    double H[T_TEST * M_TEST * N_H];              /* hidden states                  */
    double LOGITS[M_TEST * NS], P[M_TEST * NS];
    int    Y[M_TEST];                             /* targets: the first symbol      */
    double DLOGITS[M_TOY * NS];
    double DH[T_TOY * M_TOY * N_H], DZ[T_TOY * M_TOY * N4], DC[T_TOY * M_TOY * N_H];
} Work;

/* T x m latch sequences, drawn sequence by sequence, each left to right
 * (python/lstm.py draw_latch): step 0 a signal, uniform over 0..NS-1; every
 * later step a distractor, uniform over NS..V-1. Then one-hot encoded. */
static void draw_batch(NgRng *r, Work *w, int T, int m) {
    for (int i = 0; i < m; i++) {
        w->S[i] = (int)(NS * ng_rng_uniform(r));
        for (int t = 1; t < T; t++)
            w->S[t * m + i] = NS + (int)(ND * ng_rng_uniform(r));
    }
    memset(w->X, 0, sizeof(double) * (size_t)(T * m * V));
    for (int j = 0; j < T * m; j++) w->X[j * V + w->S[j]] = 1.0;
}

/* One network: the twin decides the layer and the backward pass. The RNN
 * and LSTM parameters live side by side; a twin only touches its own. */
typedef struct {
    Twin twin;
    RnnParams rp, rg, rm, rv;                     /* params, grads, Adam's moments */
    LstmParams lp, lg, lm, lv;
} Net;

static double *params(Net *n) { return n->twin == TWIN_RNN ? (double *)&n->rp : (double *)&n->lp; }
static double *grads(Net *n)  { return n->twin == TWIN_RNN ? (double *)&n->rg : (double *)&n->lg; }
static int n_params(const Net *n) {
    return (int)((n->twin == TWIN_RNN ? sizeof(RnnParams) : sizeof(LstmParams)) / sizeof(double));
}

/* The forward pass: the recurrence over all T steps, then the read-out of the
 * LAST state and its loss against the first symbol. */
static double forward(Net *n, Work *w, int T, int m) {
    const double *Why, *by;
    if (n->twin == TWIN_RNN) {
        ng_rnn_forward(w->X, NULL, n->rp.Wxh, n->rp.Whh, n->rp.bh, w->H, T, m, V, N_H);
        Why = n->rp.Why; by = n->rp.by;
    } else {
        ng_lstm_forward(w->X, NULL, NULL, n->lp.Wx, n->lp.Wh, n->lp.b,
                        w->G, w->C, w->H, T, m, V, N_H);
        Why = n->lp.Why; by = n->lp.by;
    }
    for (int i = 0; i < m; i++) w->Y[i] = w->S[i];
    ng_linear_forward(w->H + (T - 1) * m * N_H, Why, by, w->LOGITS, m, N_H, NS);
    return ng_softmax_ce_forward(w->LOGITS, w->Y, w->P, m, NS);
}

/* Backward, after a training-mode forward: through the read-out into dH at the
 * last step (zero everywhere else -- there is no other error signal), then
 * through time. The carousel twin passes cell_only = 1. */
static void backward(Net *n, Work *w) {
    const int T = T_TOY, m = M_TOY;
    double *hlast = w->H + (T - 1) * m * N_H, *dhlast = w->DH + (T - 1) * m * N_H;
    ng_softmax_ce_backward(w->P, w->Y, w->DLOGITS, m, NS);
    memset(w->DH, 0, sizeof w->DH);
    if (n->twin == TWIN_RNN) {
        ng_linear_backward(hlast, n->rp.Why, w->DLOGITS, dhlast, n->rg.Why, n->rg.by, m, N_H, NS);
        ng_rnn_backward(w->X, NULL, w->H, n->rp.Wxh, n->rp.Whh, w->DH, w->DZ, NULL, NULL,
                        n->rg.Wxh, n->rg.Whh, n->rg.bh, T, m, V, N_H, 0);
    } else {
        ng_linear_backward(hlast, n->lp.Why, w->DLOGITS, dhlast, n->lg.Why, n->lg.by, m, N_H, NS);
        ng_lstm_backward(w->X, NULL, NULL, w->G, w->C, w->H, n->lp.Wx, n->lp.Wh, w->DH,
                         w->DZ, w->DC, NULL, NULL, NULL, n->lg.Wx, n->lg.Wh, n->lg.b,
                         T, m, V, N_H, 0, n->twin == TWIN_CAROUSEL);
    }
}

static double accuracy(const Work *w, int m) {
    int correct = 0;
    for (int i = 0; i < m; i++) {
        int best = 0;                                 /* argmax, first max wins */
        for (int c = 1; c < NS; c++)
            if (w->P[i * NS + c] > w->P[i * NS + best]) best = c;
        if (best == w->Y[i]) correct++;
    }
    return (double)correct / (double)m;
}

/* Xavier init (one stream): input weights, recurrent weights, Why; biases 0,
 * except the LSTM's input and forget gate blocks. */
static void init_net(Net *n, Twin twin) {
    memset(n, 0, sizeof *n);
    n->twin = twin;
    NgRng rw;
    ng_rng_seed(&rw, TOY_SEED);
    if (twin == TWIN_RNN) {
        ng_init_linear(n->rp.Wxh, n->rp.bh, V, N_H, &rw, NG_INIT_XAVIER);
        ng_init_linear(n->rp.Whh, NULL, N_H, N_H, &rw, NG_INIT_XAVIER);
        ng_init_linear(n->rp.Why, n->rp.by, N_H, NS, &rw, NG_INIT_XAVIER);
        return;
    }
    ng_init_linear(n->lp.Wx, n->lp.b, V, N4, &rw, NG_INIT_XAVIER);
    ng_init_linear(n->lp.Wh, NULL, N_H, N4, &rw, NG_INIT_XAVIER);
    ng_init_linear(n->lp.Why, n->lp.by, N_H, NS, &rw, NG_INIT_XAVIER);
    /* forget1: the textbook biases. The others: shut to new input (i = sigmoid(-5)
     * = 0.007), open to the past (f = sigmoid(5) = 0.993: a memory of ~150 steps). */
    const double b_i = twin == TWIN_FORGET1 ? 0.0 : -5.0;
    const double b_f = twin == TWIN_FORGET1 ? 1.0 : 5.0;
    for (int j = 0; j < N_H; j++) {
        n->lp.b[j] = b_i;
        n->lp.b[N_H + j] = b_f;
    }
}

/* Train one twin from TOY_SEED's weights and print its FINAL line. n and w
 * are scratch space owned by the caller. */
static void run_toy(Twin twin, Net *n, Work *w) {
    init_net(n, twin);
    const char *how[] = {"Module 06's RNN, full BPTT",
                         "LSTM, forget bias +5, input bias -5, full BPTT",
                         "the same LSTM, gradient along the cell only (1997)",
                         "the same LSTM weights, forget bias 1, input bias 0"};
    printf("[toy %-8s] the latch, T=%d; %s; Adam lr=%g, clip %g\n",
           TWIN_NAME[twin], T_TOY, how[twin], TOY_LR, CLIP);

    double *p = params(n), *g = grads(n);
    double *am = twin == TWIN_RNN ? (double *)&n->rm : (double *)&n->lm;
    double *av = twin == TWIN_RNN ? (double *)&n->rv : (double *)&n->lv;
    const int np = n_params(n);
    NgRng rd;
    ng_rng_seed(&rd, DATA_SEED);
    double b1_pow = 1.0, b2_pow = 1.0, batch_loss = 0.0, norm = 0.0;
    int n_clip = 0;
    for (int t = 0; t < TOY_STEPS; t++) {
        draw_batch(&rd, w, T_TOY, M_TOY);            /* a fresh batch every step */
        batch_loss = forward(n, w, T_TOY, M_TOY);
        backward(n, w);
        norm = ng_clip_grad_norm(g, np, CLIP);
        if (norm > CLIP) n_clip++;
        b1_pow *= ADAM_B1;                            /* bias corrections, by product */
        b2_pow *= ADAM_B2;
        ng_adam_step(p, g, am, av, np, TOY_LR, ADAM_B1, ADAM_B2, ADAM_EPS, 0.0,
                     1.0 - b1_pow, 1.0 - b2_pow);
        if (t % (TOY_STEPS / 4) == 0 || t == TOY_STEPS - 1)
            printf("  step %4d  batch_loss=%.6f  |grad|=%.4f\n", t, batch_loss, norm);
    }

    /* The tests: fresh sequences of the training length, then four times longer. */
    NgRng rt;
    ng_rng_seed(&rt, TEST_SEED);
    draw_batch(&rt, w, T_TOY, M_TEST);
    const double loss = forward(n, w, T_TOY, M_TEST);
    const double acc = accuracy(w, M_TEST);
    draw_batch(&rt, w, T_TEST, M_TEST);
    forward(n, w, T_TEST, M_TEST);
    const double acc4 = accuracy(w, M_TEST);

    double wsum = 0.0;                                /* every parameter, canonical order */
    for (int j = 0; j < np; j++) wsum += p[j];
    const double win_00 = twin == TWIN_RNN ? n->rp.Wxh[0] : n->lp.Wx[0];
    const double wrec_00 = twin == TWIN_RNN ? n->rp.Whh[0] : n->lp.Wh[0];
    const double why_00 = twin == TWIN_RNN ? n->rp.Why[0] : n->lp.Why[0];

    printf("FINAL %s loss=%.17g acc=%.17g acc4=%.17g batch_loss=%.17g gnorm=%.17g n_clip=%d "
           "wsum=%.17g win_00=%.17g wrec_00=%.17g why_00=%.17g\n",
           TWIN_NAME[twin], loss, acc, acc4, batch_loss, norm, n_clip, wsum,
           win_00, wrec_00, why_00);
}

/* ---- Fixed-tensor check of the library ------------------------------------ */

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

/* The LSTM on (T, m, n_in, n_h) = (5, 3, 4, 6) with an initial state and
 * cell: forward, full-BPTT backward, then a backward truncated every 2 steps
 * and a cell-only backward -- python/lstm.py lstm_fingerprint(). */
static void lstm_check(void) {
    enum { T = 5, M = 3, NI = 4, NH = 6, NG = 4 * NH, NX = T * M * NI, NHS = T * M * NH };
    double Wx[NI * NG], Wh[NH * NG], b[NG], X[NX], h0[M * NH], c0[M * NH], dH[NHS];
    double G[T * M * NG], C[NHS], H[NHS], dZ[T * M * NG], dC[NHS], dX[NX];
    double dh0[M * NH], dc0[M * NH], dWx[NI * NG], dWh[NH * NG], db[NG];
    NgRng r;
    ng_rng_seed(&r, 6);
    draw(&r, Wx, NI * NG, 0.6, 0.0);
    draw(&r, Wh, NH * NG, 0.6, 0.0);
    draw(&r, b, NG, 0.5, 0.3);
    draw(&r, X, NX, 1.5, 0.0);
    draw(&r, h0, M * NH, 0.8, 0.0);
    draw(&r, c0, M * NH, 1.2, 0.0);
    draw(&r, dH, NHS, 1.0, 0.0);
    ng_lstm_forward(X, h0, c0, Wx, Wh, b, G, C, H, T, M, NI, NH);
    ng_lstm_backward(X, h0, c0, G, C, H, Wx, Wh, dH, dZ, dC, dX, dh0, dc0,
                     dWx, dWh, db, T, M, NI, NH, 0, 0);
    printf("FINAL cell");
    sums("h", H, NHS);
    sums("c", C, NHS);
    sums("dx", dX, NX);
    sums("dh0", dh0, M * NH);
    sums("dc0", dc0, M * NH);
    sums("dwx", dWx, NI * NG);
    sums("dwh", dWh, NH * NG);
    sums("db", db, NG);
    sums("dz", dZ, T * M * NG);
    sums("dc", dC, NHS);
    ng_lstm_backward(X, h0, c0, G, C, H, Wx, Wh, dH, dZ, dC, NULL, NULL, NULL,
                     dWx, dWh, db, T, M, NI, NH, 2, 0);
    sums("dwh2", dWh, NH * NG);
    sums("dc2", dC, NHS);
    ng_lstm_backward(X, h0, c0, G, C, H, Wx, Wh, dH, dZ, dC, NULL, NULL, dc0,
                     dWx, dWh, db, T, M, NI, NH, 0, 1);
    sums("dwh3", dWh, NH * NG);
    sums("dc3", dC, NHS);
    sums("dc03", dc0, M * NH);
    printf("\n");
}

int main(void) {
    Net *n = malloc(sizeof *n);
    Work *w = malloc(sizeof *w);
    if (!n || !w) {
        fprintf(stderr, "out of memory\n");
        free(n);
        free(w);
        return 1;
    }
    run_toy(TWIN_RNN, n, w);       /* Module 06's network: it cannot hold the signal */
    run_toy(TWIN_LSTM, n, w);      /* the gates, shut to input and open to the past  */
    run_toy(TWIN_CAROUSEL, n, w);  /* the same weights, the 1997 cell-only gradient  */
    run_toy(TWIN_FORGET1, n, w);   /* the same weights, the textbook gate biases     */
    lstm_check();
    free(n);
    free(w);
    return 0;
}
