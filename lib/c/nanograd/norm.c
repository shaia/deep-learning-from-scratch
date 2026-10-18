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

/* ---- Batch normalization (Module 05) ------------------------------------
 * Mirrors lib/python/nanograd/norm.py's BatchNorm (see its comment for the
 * derivation). Layout X[(i*C + c)*HW + s]: HW = 1 for a Linear's (m, C)
 * output, HW = H*W for a conv's NCHW map -- either way channel c's m = n*HW
 * values are normalized together. Sums run i, s ascending from 0.0, and every
 * expression is grouped as in the Python, so a seeded C run and the Python
 * mirror agree to the bit.
 *
 * training != 0: batch statistics; XHAT and invstd[C] are cached for
 *   backward and the running averages are updated (momentum = weight of the
 *   newest batch; unbiased variance, the paper's inference estimate).
 * training == 0: the running statistics, untouched; XHAT and invstd are
 *   still filled, so backward works in either mode. */
void ng_batchnorm_forward(const double *X, const double *gamma, const double *beta,
                          double *Y, double *XHAT, double *invstd,
                          double *run_mean, double *run_var,
                          int n, int C, int HW, double eps, double momentum,
                          int training) {
    const int m = n * HW;
    for (int c = 0; c < C; c++) {
        double mean, var;
        if (training) {
            double s = 0.0;
            for (int i = 0; i < n; i++)
                for (int k = 0; k < HW; k++)
                    s += X[(i * C + c) * HW + k];
            mean = s / (double)m;
            double s2 = 0.0;
            for (int i = 0; i < n; i++)
                for (int k = 0; k < HW; k++) {
                    const double d = X[(i * C + c) * HW + k] - mean;
                    s2 += d * d;                         /* biased: this batch */
                }
            var = s2 / (double)m;
            run_mean[c] = (1.0 - momentum) * run_mean[c] + momentum * mean;
            run_var[c] = (1.0 - momentum) * run_var[c]
                       + momentum * (var * (double)m / (double)(m - 1));
        } else {
            mean = run_mean[c];
            var = run_var[c];
        }
        invstd[c] = 1.0 / sqrt(var + eps);
        for (int i = 0; i < n; i++)
            for (int k = 0; k < HW; k++) {
                const int e = (i * C + c) * HW + k;
                XHAT[e] = (X[e] - mean) * invstd[c];
                Y[e] = gamma[c] * XHAT[e] + beta[c];
            }
    }
}

/* dbeta = sum dY, dgamma = sum dY*xhat, and -- in training mode, where every
 * x moved the batch's mean and variance --
 *     dX = (gamma*invstd/m) * (m*dY - dbeta - xhat*dgamma);
 * in evaluation mode the statistics were constants: dX = dY*gamma*invstd. */
void ng_batchnorm_backward(const double *XHAT, const double *gamma,
                           const double *invstd, const double *dY,
                           double *dX, double *dgamma, double *dbeta,
                           int n, int C, int HW, int training) {
    const int m = n * HW;
    for (int c = 0; c < C; c++) {
        double sb = 0.0, sg = 0.0;
        for (int i = 0; i < n; i++)
            for (int k = 0; k < HW; k++) {
                const int e = (i * C + c) * HW + k;
                sb += dY[e];
                sg += dY[e] * XHAT[e];
            }
        dbeta[c] = sb;
        dgamma[c] = sg;
        if (!dX) continue;
        const double kc = gamma[c] * invstd[c] / (double)m;
        for (int i = 0; i < n; i++)
            for (int k = 0; k < HW; k++) {
                const int e = (i * C + c) * HW + k;
                dX[e] = training
                    ? kc * ((double)m * dY[e] - sb - XHAT[e] * sg)
                    : dY[e] * gamma[c] * invstd[c];
            }
    }
}
