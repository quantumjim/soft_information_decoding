import numpy as np
from scipy.signal import fftconvolve


def xy(iq):
    return np.stack([np.real(iq), np.imag(iq)], -1)


class Readout:
    """KDEs f0, f1 of one qubit's calibration IQ clouds, tabulated on a grid (paper App. D, G).

    Calling it on IQ points returns (z_hat, p_soft): maximum-likelihood outcome (Eq. 12) and soft flip
    probability (Eq. 17), with p_soft = 0.5 for leaked points (Sec. II F). IQ is z-scored with the
    calibration statistics. Bandwidth 0.6 and the leakage rule (both densities < 0.01) are those of the
    code behind the published results, which differ slightly from the paper text (see README).
    """

    def __init__(self, iq0, iq1, bandwidth=0.6, leak_density=0.01, n=400):
        a0, a1 = xy(iq0), xy(iq1)
        both = np.vstack([a0, a1])
        self.mean, self.std = both.mean(0), both.std(0)
        s0, s1 = self.scale(iq0), self.scale(iq1)
        self.n, self.lo, self.hi = n, np.vstack([s0, s1]).min(0) - 2, np.vstack([s0, s1]).max(0) + 2
        self.step = (self.hi - self.lo) / n
        self.leak_density = leak_density
        self.f = [self.kde(s, bandwidth) for s in (s0, s1)]

    def scale(self, iq):
        return (xy(iq) - self.mean) / self.std

    def kde(self, pts, h):
        counts = np.histogram2d(*pts.T, bins=self.n, range=list(zip(self.lo, self.hi)))[0]
        r = np.ceil(h / self.step).astype(int)
        x, y = np.meshgrid(*(np.arange(-k, k + 1) * s for k, s in zip(r, self.step)), indexing="ij")
        u2 = (x**2 + y**2) / h**2
        return np.clip(fftconvolve(counts, np.where(u2 < 1, 2 / (np.pi * h**2) * (1 - u2), 0), mode="same"), 0, None) / len(pts)

    def lookup(self, f, s):
        i = np.floor((s - self.lo) / self.step).astype(int)
        inside = np.all((i >= 0) & (i < f.shape), axis=-1)
        i = np.clip(i, 0, np.array(f.shape) - 1)
        return np.where(inside, f[i[..., 0], i[..., 1]], 0.0)

    def leaked(self, iq):
        f = [self.lookup(f, self.scale(iq)) for f in self.f]
        return np.maximum(*f) < self.leak_density

    def __call__(self, iq, leakage=True):
        f0, f1 = (self.lookup(f, self.scale(iq)) for f in self.f)
        tot = f0 + f1
        p = np.where(tot > 0, np.minimum(f0, f1) / np.where(tot > 0, tot, 1), 0.5)
        if leakage:
            p = np.where(np.maximum(f0, f1) < self.leak_density, 0.5, p)
        return (f1 > f0).astype(np.uint8), p


def postselect(first0, second0, first1, second1):
    """First-measurement IQ of a double-measurement calibration without shots that are wrong in both
    measurements (hard flips, paper Tab. II); labels from the nearest class mean."""
    m0, m1 = np.mean(first0), np.mean(first1)
    is1 = [np.abs(a - m1) < np.abs(a - m0) for a in (first0, second0, first1, second1)]
    return first0[~(is1[0] & is1[1])], first1[is1[2] | is1[3]]
