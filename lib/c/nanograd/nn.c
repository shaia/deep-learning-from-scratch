/* nanograd layers -- Linear, activations, and fused softmax cross-entropy.
 * Each backward is the chain rule for one op, written out. Summation orders
 * match lib/python/nanograd so the C<->Python agreement test holds bit-for-bit.
 * All arrays are row-major and caller-owned. */
#include "nanograd.h"

#include <math.h>

/* ---- Linear: Y = X W + b ------------------------------------------------- */
void ng_linear_forward(const double *X, const double *W, const double *b,
                       double *Y, int m, int n_in, int n_out) {
    for (int i = 0; i < m; i++)
        for (int o = 0; o < n_out; o++) {
            double s = b[o];
            for (int k = 0; k < n_in; k++)
                s += X[i * n_in + k] * W[k * n_out + o];   /* sum over inputs */
            Y[i * n_out + o] = s;
        }
}

/* dW = X^T dY, db = column sums of dY, dX = dY W^T. All sums run over the batch
 * (or over outputs, for dX) in ascending index order -- the same order the
 * Python per-sample accumulation produces, so the last bit agrees. */
void ng_linear_backward(const double *X, const double *W, const double *dY,
                        double *dX, double *dW, double *db,
                        int m, int n_in, int n_out) {
    for (int k = 0; k < n_in; k++)
        for (int o = 0; o < n_out; o++) {
            double s = 0.0;
            for (int i = 0; i < m; i++)
                s += X[i * n_in + k] * dY[i * n_out + o];  /* dL/dW_ko */
            dW[k * n_out + o] = s;
        }
    for (int o = 0; o < n_out; o++) {
        double s = 0.0;
        for (int i = 0; i < m; i++)
            s += dY[i * n_out + o];                        /* dL/db_o */
        db[o] = s;
    }
    if (dX) {
        for (int i = 0; i < m; i++)
            for (int k = 0; k < n_in; k++) {
                double s = 0.0;
                for (int o = 0; o < n_out; o++)
                    s += dY[i * n_out + o] * W[k * n_out + o];  /* dL/dX_ik */
                dX[i * n_in + k] = s;
            }
    }
}

/* ---- Activations (elementwise) ------------------------------------------- */
void ng_relu_forward(const double *Z, double *A, int n) {
    for (int i = 0; i < n; i++) A[i] = Z[i] > 0.0 ? Z[i] : 0.0;
}
void ng_relu_backward(const double *Z, const double *dA, double *dZ, int n) {
    for (int i = 0; i < n; i++) dZ[i] = Z[i] > 0.0 ? dA[i] : 0.0;
}

void ng_tanh_forward(const double *Z, double *A, int n) {
    for (int i = 0; i < n; i++) A[i] = tanh(Z[i]);
}
void ng_tanh_backward(const double *A, const double *dA, double *dZ, int n) {
    for (int i = 0; i < n; i++) dZ[i] = dA[i] * (1.0 - A[i] * A[i]);
}

/* Overflow-safe logistic sigmoid (branch on the sign of z). */
static double sigmoid_scalar(double z) {
    if (z >= 0.0) return 1.0 / (1.0 + exp(-z));
    double ez = exp(z);
    return ez / (1.0 + ez);
}
void ng_sigmoid_forward(const double *Z, double *A, int n) {
    for (int i = 0; i < n; i++) A[i] = sigmoid_scalar(Z[i]);
}
void ng_sigmoid_backward(const double *A, const double *dA, double *dZ, int n) {
    for (int i = 0; i < n; i++) dZ[i] = dA[i] * A[i] * (1.0 - A[i]);
}

/* ---- Softmax + cross-entropy (fused, row-wise) --------------------------- */
double ng_softmax_ce_forward(const double *logits, const int *y,
                             double *P, int m, int k) {
    double total = 0.0;
    for (int i = 0; i < m; i++) {
        const double *row = logits + i * k;
        double mx = row[0];
        for (int c = 1; c < k; c++) if (row[c] > mx) mx = row[c];  /* stabilize */
        double s = 0.0;
        for (int c = 0; c < k; c++) {
            double e = exp(row[c] - mx);
            P[i * k + c] = e;
            s += e;
        }
        for (int c = 0; c < k; c++) P[i * k + c] /= s;
        total += -log(P[i * k + y[i]] + 1e-12);
    }
    return total / (double)m;
}

/* dL/dlogits = (P - onehot(y)) / m -- the clean form fusing buys us. */
void ng_softmax_ce_backward(const double *P, const int *y,
                            double *dlogits, int m, int k) {
    for (int i = 0; i < m; i++)
        for (int c = 0; c < k; c++)
            dlogits[i * k + c] = (P[i * k + c] - (c == y[i] ? 1.0 : 0.0))
                                 / (double)m;
}
