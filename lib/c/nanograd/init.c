/* nanograd weight initialization -- Xavier / He / small-uniform.
 * Draws from the RNG in row-major order and matches lib/python/nanograd/init.py
 * bit-for-bit (same std, same draw order), so a seeded C net equals the Python one. */
#include "nanograd.h"

#include <math.h>

void ng_init_linear(double *W, double *b, int n_in, int n_out,
                    NgRng *r, NgInit kind) {
    if (kind == NG_INIT_SMALL) {
        /* Module-01 baseline: uniform [-1, 1). */
        for (int i = 0; i < n_in; i++)
            for (int o = 0; o < n_out; o++)
                W[i * n_out + o] = ng_rng_signed(r);
    } else {
        /* Fan-in-scaled Gaussian: He (2/fan_in) or Xavier (1/fan_in). */
        double std = (kind == NG_INIT_HE) ? sqrt(2.0 / (double)n_in)
                                          : sqrt(1.0 / (double)n_in);
        for (int i = 0; i < n_in; i++)
            for (int o = 0; o < n_out; o++)
                W[i * n_out + o] = std * ng_rng_normal(r);
    }
    for (int o = 0; o < n_out; o++) b[o] = 0.0;   /* biases start at zero */
}
