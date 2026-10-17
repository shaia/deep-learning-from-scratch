/* nanograd data augmentation -- AlexNet's crops and mirrors (Module 04).
 *
 * Mirrors lib/python/nanograd/augment.py. Training images are shown as a
 * random crop x crop window, left-right mirrored half the time; evaluation uses
 * the fixed center crop. A shifted or mirrored object is still the same object,
 * so every variant is a free, correctly labeled extra example.
 *
 * Per image, three uniforms are drawn in order -- row offset, column offset,
 * flip -- which is exactly the row-major (n, 3) block the Python version draws
 * with rng.random((n, 3)), so a seeded C run crops and flips identically. */
#include "nanograd.h"

/* X:[n,C,H,W] -> out:[n,C,crop,crop]. Offsets are uniform over the H-crop+1
 * (resp. W-crop+1) valid positions; `flip` = 0 disables mirroring (the flip
 * uniform is still drawn, so the stream stays aligned either way). */
void ng_random_crop_flip(const double *X, double *out,
                         int n, int C, int H, int W, int crop, int flip,
                         NgRng *r) {
    for (int i = 0; i < n; i++) {
        const int dy = (int)(ng_rng_uniform(r) * (double)(H - crop + 1));
        const int dx = (int)(ng_rng_uniform(r) * (double)(W - crop + 1));
        const double u_flip = ng_rng_uniform(r);  /* drawn even if !flip: */
        const int mirror = flip && u_flip < 0.5;  /* no short-circuit skip */
        for (int c = 0; c < C; c++)
            for (int h = 0; h < crop; h++)
                for (int w = 0; w < crop; w++) {
                    const int src_w = mirror ? crop - 1 - w : w;
                    out[((i * C + c) * crop + h) * crop + w] =
                        X[((i * C + c) * H + (dy + h)) * W + (dx + src_w)];
                }
    }
}

/* The deterministic evaluation view: the middle crop x crop window. */
void ng_center_crop(const double *X, double *out,
                    int n, int C, int H, int W, int crop) {
    const int oy = (H - crop) / 2, ox = (W - crop) / 2;
    for (int i = 0; i < n; i++)
        for (int c = 0; c < C; c++)
            for (int h = 0; h < crop; h++)
                for (int w = 0; w < crop; w++)
                    out[((i * C + c) * crop + h) * crop + w] =
                        X[((i * C + c) * H + (oy + h)) * W + (ox + w)];
}
