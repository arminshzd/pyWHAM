"""Numerical Recipes utilities translated from the C implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Ran2Generator:
    """Stateful reimplementation of the ``ran2`` random number generator."""

    seed: int
    idum2: int = 123456789
    iy: int = 0
    iv: List[int] = field(default_factory=lambda: [0] * 32)

    def random(self) -> float:
        IM1 = 2147483563
        IM2 = 2147483399
        AM = 1.0 / IM1
        IMM1 = IM1 - 1
        IA1 = 40014
        IA2 = 40692
        IQ1 = 53668
        IQ2 = 52774
        IR1 = 12211
        IR2 = 3791
        NTAB = 32
        NDIV = 1 + IMM1 // NTAB
        RNMX = 1.0 - 2.220446049250313e-16

        if self.seed <= 0:
            if -self.seed < 1:
                self.seed = 1
            else:
                self.seed = -self.seed
            self.idum2 = self.seed
            for j in range(NTAB + 7, -1, -1):
                k = self.seed // IQ1
                self.seed = IA1 * (self.seed - k * IQ1) - k * IR1
                if self.seed < 0:
                    self.seed += IM1
                if j < NTAB:
                    self.iv[j] = self.seed
            self.iy = self.iv[0]

        k = self.seed // IQ1
        self.seed = IA1 * (self.seed - k * IQ1) - k * IR1
        if self.seed < 0:
            self.seed += IM1

        k = self.idum2 // IQ2
        self.idum2 = IA2 * (self.idum2 - k * IQ2) - k * IR2
        if self.idum2 < 0:
            self.idum2 += IM2

        j = self.iy // NDIV
        self.iy = self.iv[j] - self.idum2
        self.iv[j] = self.seed
        if self.iy < 1:
            self.iy += IMM1

        temp = AM * self.iy
        if temp > RNMX:
            return RNMX
        return temp


def locate(xx: List[float], x: float) -> int:
    """Locate the index of ``x`` within the sorted array ``xx``.

    This matches the zero-based behavior of the C helper used by the WHAM
    bootstrap routines.
    """

    n = len(xx) - 1
    jl = 0
    ju = n + 1
    ascnd = xx[n] > xx[0]

    while ju - jl > 1:
        jm = (ju + jl) >> 1
        if (x > xx[jm]) == ascnd:
            jl = jm
        else:
            ju = jm

    while jl > 0 and jl != n and xx[jl] <= xx[jl - 1]:
        jl -= 1

    return jl
