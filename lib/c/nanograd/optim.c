/* nanograd optimizers -- SGD(+momentum), RMSProp, Adam. Elementwise over a flat
 * parameter array; state buffers are caller-owned. Formulas mirror
 * lib/python/nanograd/optim.py exactly. */
#include "nanograd.h"

#include <math.h>

/* theta <- theta - lr * v,  v <- momentum*v + g   (momentum=0 => plain SGD). */
void ng_sgd_step(double *p, const double *g, double *vel, int n,
                 double lr, double momentum, double weight_decay) {
    for (int i = 0; i < n; i++) {
        double gi = g[i];
        if (weight_decay != 0.0) gi += weight_decay * p[i];   /* L2 */
        vel[i] = momentum * vel[i] + gi;
        p[i] -= lr * vel[i];
    }
}

/* s <- beta*s + (1-beta)*g^2 ;  theta <- theta - lr*g/(sqrt(s)+eps). */
void ng_rmsprop_step(double *p, const double *g, double *s, int n,
                     double lr, double beta, double eps, double weight_decay) {
    for (int i = 0; i < n; i++) {
        double gi = g[i];
        if (weight_decay != 0.0) gi += weight_decay * p[i];
        s[i] = beta * s[i] + (1.0 - beta) * (gi * gi);
        p[i] -= lr * gi / (sqrt(s[i]) + eps);
    }
}

/* Adam. b1c = 1-beta1^t and b2c = 1-beta2^t are supplied by the caller (tracked
 * by repeated multiplication) so C and Python never disagree on a pow() ulp. */
void ng_adam_step(double *p, const double *g, double *m, double *v, int n,
                  double lr, double beta1, double beta2, double eps,
                  double weight_decay, double b1c, double b2c) {
    for (int i = 0; i < n; i++) {
        double gi = g[i];
        if (weight_decay != 0.0) gi += weight_decay * p[i];
        m[i] = beta1 * m[i] + (1.0 - beta1) * gi;
        v[i] = beta2 * v[i] + (1.0 - beta2) * (gi * gi);
        double m_hat = m[i] / b1c;
        double v_hat = v[i] / b2c;
        p[i] -= lr * m_hat / (sqrt(v_hat) + eps);
    }
}
