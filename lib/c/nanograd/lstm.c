/* nanograd long short-term memory (Module 07) -- an LSTM layer with forget
 * gate over a whole sequence, and backpropagation through time along both of
 * its paths, the hidden state and the cell.
 *
 *   z<t> = x<t> Wx + h<t-1> Wh + b        four blocks side by side: [i | f | g | o]
 *   i, f, o = sigmoid(z_i, z_f, z_o)      g = tanh(z_g)
 *   c<t> = f * c<t-1> + i * g             h<t> = o * tanh(c<t>)
 *
 * Layout is time-major, row-major within a step, as rnn.c: X[(t*m + i)*n_in +
 * k], H and C [(t*m + i)*n_h + j], the activated gates G and their gradients
 * dZ [(t*m + i)*4n_h + o]. Weights are Wx [k*4n_h + o], Wh [j*4n_h + o]. Every
 * sum runs in ascending index order inside a step, and the shared weights'
 * gradients accumulate over time in DESCENDING t (the order BPTT walks), so
 * the C-order mirror in topics/07-lstm/python/lstm.py can reproduce them to
 * the last bit. All arrays are caller-owned; nothing is allocated here. */
#include "nanograd.h"

#include <math.h>

/* nn.c's overflow-safe logistic sigmoid (branch on the sign of z). */
static double sigmoid(double z) {
    if (z >= 0.0) return 1.0 / (1.0 + exp(-z));
    double ez = exp(z);
    return ez / (1.0 + ez);
}

/* Forward over all T steps. h0 and c0 (m x n_h) are the state and cell before
 * step 0; NULL means zeros, and their terms are then skipped rather than
 * added. For each gate pre-activation: start from the bias, add the input
 * terms (k ascending), then the recurrent terms (j ascending), then squash --
 * tanh for the candidate block, the sigmoid for the three gates. G keeps the
 * activated gates, C the cells; backward needs both. */
void ng_lstm_forward(const double *X, const double *h0, const double *c0,
                     const double *Wx, const double *Wh, const double *b,
                     double *G, double *C, double *H, int T, int m, int n_in, int n_h) {
    const int n4 = 4 * n_h;
    for (int t = 0; t < T; t++) {
        const double *x = X + (long)t * m * n_in;
        const double *hp = t > 0 ? H + (long)(t - 1) * m * n_h : h0;   /* h<t-1> */
        const double *cp = t > 0 ? C + (long)(t - 1) * m * n_h : c0;   /* c<t-1> */
        double *g = G + (long)t * m * n4;
        double *c = C + (long)t * m * n_h;
        double *h = H + (long)t * m * n_h;
        for (int i = 0; i < m; i++)
            for (int o = 0; o < n4; o++) {
                double s = b[o];
                for (int k = 0; k < n_in; k++)
                    s += x[i * n_in + k] * Wx[k * n4 + o];           /* what came in    */
                if (hp)
                    for (int j = 0; j < n_h; j++)
                        s += hp[i * n_h + j] * Wh[j * n4 + o];       /* what it showed  */
                const int candidate = o >= 2 * n_h && o < 3 * n_h;
                g[i * n4 + o] = candidate ? tanh(s) : sigmoid(s);
            }
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n_h; j++) {
                const double *gr = g + i * n4;
                const double gi = gr[j], gf = gr[n_h + j], gg = gr[2 * n_h + j], go = gr[3 * n_h + j];
                const double write = gi * gg;
                /* keep some of the old cell, write some of the new */
                const double cv = cp ? gf * cp[i * n_h + j] + write : write;
                c[i * n_h + j] = cv;
                h[i * n_h + j] = go * tanh(cv);                      /* show some of it */
            }
    }
}

/* Backpropagation through time, after ng_lstm_forward. dH (T x m x n_h) is
 * dL/dh<t> from whatever read each step's state. Walking t from T-1 down to 0,
 * per element:
 *
 *   dh<t> = dH<t> + dz<t+1> Wh^T                       (o ascending; the h path)
 *   dc<t> = dh<t> o (1 - tanh(c<t>)^2) + dc<t+1> f<t+1>    (the carousel)
 *   dz_i = dc g i(1-i)   dz_f = dc c<t-1> f(1-f)   dz_g = dc i (1-g^2)
 *   dz_o = dh tanh(c<t>) o(1-o)
 *
 * dZ and dC keep the gate and cell gradients of every step. Each step adds its
 * share to the shared weights' gradients, which are zeroed first: dWx +=
 * x<t>^T dz<t>, dWh += h<t-1>^T dz<t>, db += sum_i dz<t>. dX (dX<t> = dz<t>
 * Wx^T), dh0 (= dz<0> Wh^T) and dc0 (= dc<0> f<0>) may be NULL.
 *
 * truncate = k > 0 cuts BOTH paths into the past every k steps (no gradient
 * from step t back to step t-1, or to h0/c0, when t % k == 0): chunked
 * training. cell_only != 0 is Hochreiter & Schmidhuber's 1997 truncated
 * gradient: the h path is cut at every step, and the error travels back in
 * time only along the cell. */
void ng_lstm_backward(const double *X, const double *h0, const double *c0,
                      const double *G, const double *C, const double *H,
                      const double *Wx, const double *Wh, const double *dH,
                      double *dZ, double *dC, double *dX, double *dh0, double *dc0,
                      double *dWx, double *dWh, double *db,
                      int T, int m, int n_in, int n_h, int truncate, int cell_only) {
    const int n4 = 4 * n_h;
    for (int j = 0; j < n_in * n4; j++) dWx[j] = 0.0;
    for (int j = 0; j < n_h * n4; j++) dWh[j] = 0.0;
    for (int o = 0; o < n4; o++) db[o] = 0.0;

    for (int t = T - 1; t >= 0; t--) {
        const double *x = X + (long)t * m * n_in;
        const double *hp = t > 0 ? H + (long)(t - 1) * m * n_h : h0;
        const double *cp = t > 0 ? C + (long)(t - 1) * m * n_h : c0;
        const double *g = G + (long)t * m * n4;
        const double *c = C + (long)t * m * n_h;
        const double *dh_out = dH + (long)t * m * n_h;
        double *dz = dZ + (long)t * m * n4;
        double *dc = dC + (long)t * m * n_h;
        /* Does step t+1's gradient reach back to step t? Not past the end, and
         * not across a truncation cut; along h, not at all when cell_only. */
        const int flows = (t + 1 < T) && !(truncate > 0 && (t + 1) % truncate == 0);
        const double *dz_next = dZ + (long)(t + 1) * m * n4;
        const double *dc_next = dC + (long)(t + 1) * m * n_h;
        const double *g_next = G + (long)(t + 1) * m * n4;

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n_h; j++) {
                double dh = dh_out[i * n_h + j];
                if (flows && !cell_only)
                    for (int o = 0; o < n4; o++)
                        dh += dz_next[i * n4 + o] * Wh[j * n4 + o];
                const double *gr = g + i * n4;
                const double gi = gr[j], gf = gr[n_h + j], gg = gr[2 * n_h + j], go = gr[3 * n_h + j];
                const double tc = tanh(c[i * n_h + j]);
                double d = dh * go * (1.0 - tc * tc);                  /* via h<t>        */
                if (flows)
                    d += dc_next[i * n_h + j] * g_next[i * n4 + n_h + j];  /* along the cell */
                dc[i * n_h + j] = d;
                const double cpv = cp ? cp[i * n_h + j] : 0.0;
                dz[i * n4 + j]           = d * gg * (gi * (1.0 - gi));   /* input gate  */
                dz[i * n4 + n_h + j]     = d * cpv * (gf * (1.0 - gf));  /* forget gate */
                dz[i * n4 + 2 * n_h + j] = d * gi * (1.0 - gg * gg);     /* candidate   */
                dz[i * n4 + 3 * n_h + j] = dh * tc * (go * (1.0 - go));  /* output gate */
            }

        /* The shared weights collect this step's share. */
        for (int k = 0; k < n_in; k++)
            for (int o = 0; o < n4; o++) {
                double s = dWx[k * n4 + o];
                for (int i = 0; i < m; i++)
                    s += x[i * n_in + k] * dz[i * n4 + o];
                dWx[k * n4 + o] = s;
            }
        if (hp)
            for (int j = 0; j < n_h; j++)
                for (int o = 0; o < n4; o++) {
                    double s = dWh[j * n4 + o];
                    for (int i = 0; i < m; i++)
                        s += hp[i * n_h + j] * dz[i * n4 + o];
                    dWh[j * n4 + o] = s;
                }
        for (int o = 0; o < n4; o++) {
            double s = db[o];
            for (int i = 0; i < m; i++) s += dz[i * n4 + o];
            db[o] = s;
        }
        if (dX) {
            double *dx = dX + (long)t * m * n_in;
            for (int i = 0; i < m; i++)
                for (int k = 0; k < n_in; k++) {
                    double s = 0.0;
                    for (int o = 0; o < n4; o++)
                        s += dz[i * n4 + o] * Wx[k * n4 + o];
                    dx[i * n_in + k] = s;
                }
        }
    }

    /* What reaches the initial state and cell (0 % k == 0: a cut at h0/c0 too). */
    if (dh0)
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n_h; j++) {
                double s = 0.0;
                if (!(truncate > 0) && !cell_only)
                    for (int o = 0; o < n4; o++)
                        s += dZ[i * n4 + o] * Wh[j * n4 + o];
                dh0[i * n_h + j] = s;
            }
    if (dc0)
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n_h; j++)
                dc0[i * n_h + j] = truncate > 0 ? 0.0 : dC[i * n_h + j] * G[i * n4 + n_h + j];
}
