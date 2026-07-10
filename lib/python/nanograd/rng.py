"""
Deterministic RNG for nanograd -- the same 64-bit LCG used since Module 00.

Why a hand-rolled RNG in a library that already imports NumPy? Because the C
mirror (lib/c/nanograd) has to draw *the exact same numbers in the exact same
order* so the C<->Python agreement test can hold to full double precision. A
shared, dead-simple generator is the only way to guarantee that across two
languages. `numpy.random` is used for the metric-checked (non-bit-exact) paths.

`normal()` is Box-Muller on top of the uniform stream, with a one-value spare
cache -- implemented identically in rng.c -- so Gaussian weight initialization
(Xavier/He) is bit-exact across C and Python too.
"""

import math

_MASK64 = (1 << 64) - 1


class Rng:
    """64-bit linear congruential generator (the glibc/PCG multiplier).

    State is held on the instance (never a module global), so independent
    instances never interfere and the class reads like the C `Rng` struct it
    mirrors. A single instance is not synchronized -- don't share one across
    threads.
    """

    def __init__(self, seed: int):
        self.state = seed & _MASK64
        self.has_spare = False   # Box-Muller caches its second normal here
        self.spare = 0.0

    def next_u64(self) -> int:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & _MASK64
        return self.state

    def uniform(self) -> float:
        """Uniform double in [0, 1): top 53 bits of the state over 2^53."""
        return (self.next_u64() >> 11) * (1.0 / 9007199254740992.0)

    def signed(self) -> float:
        """Uniform double in [-1, 1)."""
        return 2.0 * self.uniform() - 1.0

    def normal(self) -> float:
        """Standard normal N(0, 1) via Box-Muller, caching the paired sample.

        Draw two uniforms u1, u2; then r*cos and r*sin (with r = sqrt(-2 ln u1))
        are two independent standard normals. We return the first and stash the
        second in `spare`. rng.c does exactly this, so He/Xavier init matches
        bit-for-bit. `u1` is guarded away from 0 so log() is finite.
        """
        if self.has_spare:
            self.has_spare = False
            return self.spare
        u1 = self.uniform()
        while u1 <= 0.0:            # log(0) guard; both impls redraw identically
            u1 = self.uniform()
        u2 = self.uniform()
        r = math.sqrt(-2.0 * math.log(u1))
        self.spare = r * math.sin(2.0 * math.pi * u2)
        self.has_spare = True
        return r * math.cos(2.0 * math.pi * u2)
