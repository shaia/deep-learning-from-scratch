/* nanograd normalization -- AlexNet's local response normalization (Module 04).
 *
 * Mirrors lib/python/nanograd/norm.py (see its docstring for the derivation).
 * At every (sample, row, column), each channel's activation is divided by a
 * power of the energy in its channel neighborhood:
 *
 *   s[c] = k + alpha * sum_{j=max(0,c-n/2)}^{min(C-1,c+n/2)} a[j]^2
 *   b[c] = a[c] * s[c]^(-beta)
 *
 *   da[m] = dB[m] * s[m]^-beta - 2*alpha*beta * a[m] * sum_{i in N(m)} dB[i]*b[i]/s[i]
 *
 * Channel neighbors of an NCHW element sit H*W apart in memory. Window sums run
 * j ascending from 0.0 -- the order norm.py's offset-by-offset sum produces --
 * and every product is grouped as in the Python, so the two agree to the
 * last few ulps (pow() is the only cross-library difference). */
#include "nanograd.h"

#include <math.h>

/* X, Y, S: [n,C,H,W]. S caches the per-element denominators base s[c] for
 * backward. `size` is the window width n (5 in AlexNet). */
void ng_lrn_forward(const double *X, double *Y, double *S,
                    int n, int C, int H, int W,
                    int size, double alpha, double beta, double k) {
    const int half = size / 2, plane = H * W;
    for (int i = 0; i < n; i++)
        for (int c = 0; c < C; c++)
            for (int hw = 0; hw < plane; hw++) {
                const int lo = c - half < 0 ? 0 : c - half;
                const int hi = c + half > C - 1 ? C - 1 : c + half;
                double sq = 0.0;
                for (int j = lo; j <= hi; j++) {
                    double a = X[(i * C + j) * plane + hw];
                    sq += a * a;                          /* local energy */
                }
                const int e = (i * C + c) * plane + hw;
                S[e] = k + alpha * sq;
                Y[e] = X[e] * pow(S[e], -beta);
            }
}

void ng_lrn_backward(const double *X, const double *Y, const double *S,
                     const double *dY, double *dX,
                     int n, int C, int H, int W,
                     int size, double alpha, double beta) {
    const int half = size / 2, plane = H * W;
    for (int i = 0; i < n; i++)
        for (int m = 0; m < C; m++)
            for (int hw = 0; hw < plane; hw++) {
                const int lo = m - half < 0 ? 0 : m - half;
                const int hi = m + half > C - 1 ? C - 1 : m + half;
                /* The coupling term: every b[i] whose window held a[m]. */
                double t = 0.0;
                for (int j = lo; j <= hi; j++) {
                    const int ej = (i * C + j) * plane + hw;
                    t += dY[ej] * Y[ej] / S[ej];
                }
                const int e = (i * C + m) * plane + hw;
                double direct = dY[e] * pow(S[e], -beta);  /* through a[m] itself */
                dX[e] = direct - 2.0 * alpha * beta * X[e] * t;
            }
}
