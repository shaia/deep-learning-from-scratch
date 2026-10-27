/* nanograd recurrent layer (Module 06) -- a vanilla tanh RNN over a whole
 * sequence, and backpropagation through time.
 *
 *   z<t> = x<t> Wxh + h<t-1> Whh + bh,      h<t> = tanh(z<t>)
 *
 * Layout is time-major, row-major within a step: X[(t*m + i)*n_in + k],
 * H[(t*m + i)*n_h + j] -- T steps of an (m x n) batch, stacked in time. Every
 * sum runs in ascending index order inside a step, and the shared weights'
 * gradients accumulate over time in DESCENDING t (the order BPTT walks), so
 * lib/python/nanograd's C-order mirror can reproduce them to the last bit.
 * All arrays are caller-owned; nothing is allocated here. */
#include "nanograd.h"

#include <math.h>

/* Forward over all T steps. h0 (m x n_h) is the state before step 0; NULL
 * means zeros, and its terms are then skipped rather than added. For each
 * element: start from the bias, add the input terms (k ascending), then the
 * recurrent terms (j ascending), then squash. */
void ng_rnn_forward(const double *X, const double *h0,
                    const double *Wxh, const double *Whh, const double *bh,
                    double *H, int T, int m, int n_in, int n_h) {
    for (int t = 0; t < T; t++) {
        const double *x = X + (long)t * m * n_in;
        const double *hp = t > 0 ? H + (long)(t - 1) * m * n_h : h0;  /* h<t-1> */
        double *h = H + (long)t * m * n_h;
        for (int i = 0; i < m; i++)
            for (int o = 0; o < n_h; o++) {
                double s = bh[o];
                for (int k = 0; k < n_in; k++)
                    s += x[i * n_in + k] * Wxh[k * n_h + o];     /* what came in   */
                if (hp)
                    for (int j = 0; j < n_h; j++)
                        s += hp[i * n_h + j] * Whh[j * n_h + o]; /* what it remembered */
                h[i * n_h + o] = tanh(s);
            }
    }
}

/* Backpropagation through time, after ng_rnn_forward. dH (T x m x n_h) is
 * dL/dh<t> from whatever read each step's state. Walking t from T-1 down to 0:
 *
 *   dh<t> = dH<t> + dz<t+1> Whh^T          (from the output, and from the future)
 *   dz<t> = dh<t> * (1 - h<t>^2)           (back through tanh; stored in dZ)
 *
 * and each step adds its share to the shared weights' gradients, which are
 * zeroed first: dWxh += x<t>^T dz<t>, dWhh += h<t-1>^T dz<t>, dbh += sum_i dz<t>.
 * dX (dX<t> = dz<t> Wxh^T) and dh0 (= dz<0> Whh^T) may be NULL.
 *
 * truncate = k > 0 cuts the path into the past every k steps: no gradient
 * flows from step t back to step t-1 (or to h0) when t % k == 0 -- training in
 * chunks of k with the state carried across. k = 1 is Elman's 1990 training;
 * 0 is full BPTT. */
void ng_rnn_backward(const double *X, const double *h0, const double *H,
                     const double *Wxh, const double *Whh, const double *dH,
                     double *dZ, double *dX, double *dh0,
                     double *dWxh, double *dWhh, double *dbh,
                     int T, int m, int n_in, int n_h, int truncate) {
    for (int j = 0; j < n_in * n_h; j++) dWxh[j] = 0.0;
    for (int j = 0; j < n_h * n_h; j++) dWhh[j] = 0.0;
    for (int o = 0; o < n_h; o++) dbh[o] = 0.0;

    for (int t = T - 1; t >= 0; t--) {
        const double *x = X + (long)t * m * n_in;
        const double *hp = t > 0 ? H + (long)(t - 1) * m * n_h : h0;
        const double *h = H + (long)t * m * n_h;
        const double *dh_out = dH + (long)t * m * n_h;
        double *dz = dZ + (long)t * m * n_h;
        /* Does step t+1's gradient reach back to step t? Not past the end,
         * and not across a truncation cut. */
        const int flows = (t + 1 < T) && !(truncate > 0 && (t + 1) % truncate == 0);
        const double *dz_next = dZ + (long)(t + 1) * m * n_h;

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n_h; j++) {
                double s = dh_out[i * n_h + j];
                if (flows)
                    for (int o = 0; o < n_h; o++)
                        s += dz_next[i * n_h + o] * Whh[j * n_h + o];
                const double hv = h[i * n_h + j];
                dz[i * n_h + j] = s * (1.0 - hv * hv);           /* tanh' = 1 - h^2 */
            }

        /* The shared weights collect this step's share. */
        for (int k = 0; k < n_in; k++)
            for (int o = 0; o < n_h; o++) {
                double s = dWxh[k * n_h + o];
                for (int i = 0; i < m; i++)
                    s += x[i * n_in + k] * dz[i * n_h + o];
                dWxh[k * n_h + o] = s;
            }
        if (hp)
            for (int j = 0; j < n_h; j++)
                for (int o = 0; o < n_h; o++) {
                    double s = dWhh[j * n_h + o];
                    for (int i = 0; i < m; i++)
                        s += hp[i * n_h + j] * dz[i * n_h + o];
                    dWhh[j * n_h + o] = s;
                }
        for (int o = 0; o < n_h; o++) {
            double s = dbh[o];
            for (int i = 0; i < m; i++) s += dz[i * n_h + o];
            dbh[o] = s;
        }
        if (dX) {
            double *dx = dX + (long)t * m * n_in;
            for (int i = 0; i < m; i++)
                for (int k = 0; k < n_in; k++) {
                    double s = 0.0;
                    for (int o = 0; o < n_h; o++)
                        s += dz[i * n_h + o] * Wxh[k * n_h + o];
                    dx[i * n_in + k] = s;
                }
        }
    }

    if (dh0) {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n_h; j++) {
                double s = 0.0;
                if (!(truncate > 0))                   /* 0 % k == 0: cut at h0 too */
                    for (int o = 0; o < n_h; o++)
                        s += dZ[i * n_h + o] * Whh[j * n_h + o];
                dh0[i * n_h + j] = s;
            }
    }
}
