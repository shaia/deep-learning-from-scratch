/* nanograd RNG -- 64-bit LCG + Box-Muller normal. Mirrors lib/python/nanograd/rng.py
 * exactly so weight initialization is bit-identical across C and Python. */
#include "nanograd.h"

#include <math.h>

/* pi to full double precision -- this literal rounds to the same IEEE double as
 * Python's math.pi (3.141592653589793), keeping Box-Muller bit-exact across the
 * two. We define it ourselves because M_PI is not part of standard C11. */
#define NG_PI 3.14159265358979323846

void ng_rng_seed(NgRng *r, uint64_t seed) {
    r->state = seed;
    r->has_spare = 0;
    r->spare = 0.0;
}

static uint64_t rng_next_u64(NgRng *r) {
    r->state = r->state * 6364136223846793005ULL + 1442695040888963407ULL;
    return r->state;
}

/* Uniform double in [0, 1): top 53 bits over 2^53. */
double ng_rng_uniform(NgRng *r) {
    return (double)(rng_next_u64(r) >> 11) * (1.0 / 9007199254740992.0);
}

/* Uniform double in [-1, 1). */
double ng_rng_signed(NgRng *r) { return 2.0 * ng_rng_uniform(r) - 1.0; }

/* Standard normal via Box-Muller, caching the paired sample (matches rng.py). */
double ng_rng_normal(NgRng *r) {
    if (r->has_spare) {
        r->has_spare = 0;
        return r->spare;
    }
    double u1 = ng_rng_uniform(r);
    while (u1 <= 0.0)                     /* log(0) guard; Python redraws too */
        u1 = ng_rng_uniform(r);
    double u2 = ng_rng_uniform(r);
    double rad = sqrt(-2.0 * log(u1));
    r->spare = rad * sin(2.0 * NG_PI * u2);
    r->has_spare = 1;
    return rad * cos(2.0 * NG_PI * u2);
}
