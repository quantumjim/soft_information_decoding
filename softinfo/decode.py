from dataclasses import replace
from functools import reduce

import numpy as np
import pymatching
import scipy.sparse as sp

from .circuit import repetition_code


class SoftMatcher:
    """Matching graph of a stim circuit whose measurement edges are reweighted per shot.

    The circuit must have p_soft = 0. A soft flip of measurement m flips exactly the detectors that
    read m, so every measurement owns one edge; per shot its probability is xor-combined with p_s(μ).
    Algorithm 1 and Eqs. 18 and B1 of the paper are special cases of this rule.
    """

    def __init__(self, circuit):
        assert "M(" not in str(circuit), "build the circuit with p_soft = 0"
        edges = {}
        for e in circuit.detector_error_model(decompose_errors=True).flattened():
            if e.type != "error":
                continue
            parts = [[set(), set()]]
            for t in e.targets_copy():
                if t.is_separator():
                    parts.append([set(), set()])
                else:
                    parts[-1][t.is_logical_observable_id()] ^= {t.val}
            whole = [reduce(set.__xor__, (p[i] for p in parts)) for i in (0, 1)]
            for dets, obs in [whole] if len(whole[0]) <= 2 else parts:  # stim also splits graph-like errors
                if dets:
                    key, p = (tuple(sorted(dets)), tuple(sorted(obs))), e.args_copy()[0]
                    q = edges.get(key, 0.0)
                    edges[key] = p + q - 2 * p * q
        self.m2d = circuit.compile_m2d_converter()
        n = circuit.num_measurements
        flips = [a ^ b for a, b in zip(self.m2d.convert(measurements=np.eye(n, dtype=bool), separate_observables=True),
                                       self.m2d.convert(measurements=np.zeros((1, n), bool), separate_observables=True))]
        meas_keys = [(tuple(np.flatnonzero(d)), tuple(np.flatnonzero(o))) for d, o in zip(*flips)]
        for k in meas_keys:
            edges.setdefault(k, 0.0)
        keys = list(edges)
        index = {k: i for i, k in enumerate(keys)}
        self.p_base = np.array(list(edges.values()))
        self.meas_edge = np.array([index[k] for k in meas_keys])
        self.H, self.F = (sp.csc_matrix((np.ones(len(r), np.uint8), (r, c)), shape=(rows, len(keys)))
                          for rows, (r, c) in [(circuit.num_detectors, _pairs(keys, 0)), (circuit.num_observables, _pairs(keys, 1))])

    def edge_probs(self, p_soft):
        """Edge probabilities for soft flip probabilities [..., n_measurements]."""
        p_soft = np.asarray(p_soft, float)
        q = np.broadcast_to(1 - 2 * self.p_base, p_soft.shape[:-1] + self.p_base.shape).copy()
        np.multiply.at(q, (..., self.meas_edge), 1 - 2 * p_soft)
        return np.clip((1 - q) / 2, 1e-12, 0.5)

    def matching(self, p):
        return pymatching.Matching.from_check_matrix(self.H, weights=np.log((1 - p) / p), faults_matrix=self.F,
                                                     use_virtual_boundary_node=True)

    def decode_static(self, meas, p_soft):
        """Logical error per shot with one graph for all shots; p_soft: [n_measurements]."""
        det, obs = self.m2d.convert(measurements=np.asarray(meas, bool), separate_observables=True)
        return (self.matching(self.edge_probs(p_soft)).decode_batch(det) != obs)[:, 0]

    def decode_soft(self, meas, p_soft):
        """Logical error per shot with a per-shot reweighted graph; p_soft: [shots, n_measurements]."""
        det, obs = self.m2d.convert(measurements=np.asarray(meas, bool), separate_observables=True)
        P = self.edge_probs(p_soft)
        return np.array([self.matching(P[i]).decode(det[i])[0] for i in range(len(det))]) != obs[:, 0]


def _pairs(keys, pos):
    return np.array([(j, i) for i, k in enumerate(keys) for j in k[pos]]).T


def truncate(p, bits):
    """Round p_s to 2^b uniform levels on [0, 0.5] (paper App. G)."""
    levels = 2**bits - 1
    return np.round(np.asarray(p) / 0.5 * levels) / levels * 0.5


def subset_columns(start, d, d_full, T):
    """Measurement columns of the length-d sub-chain starting at code qubit `start` (App. E)."""
    return np.r_[[t * (d_full - 1) + start + i for t in range(T) for i in range(d - 1)], T * (d_full - 1) + start + np.arange(d)]


def decode_subsets(z, ps, T, xbasis, logical, noise, distances, bits=()):
    """Logical errors summed over all sub-chains of each distance: ({d: {method: errors}}, {d: samples}).

    Methods: soft, hard (p_s = noise.p_soft) and informed (p_s = mean of ps), or with `bits` soft
    decoding with p_s truncated to b bits (64 = full precision).
    """
    d_full = (z.shape[1] + T) // (T + 1)
    errs, samples = {}, {}
    for d in distances:
        m = SoftMatcher(repetition_code(d, T, xbasis, logical, replace(noise, p_soft=0), subsampling=d < d_full))
        n = m.meas_edge.size
        errs[d] = {}
        for start in range(d_full - d + 1):
            cols = subset_columns(start, d, d_full, T)
            zc, pc = z[:, cols], ps[:, cols]
            res = ({b: m.decode_soft(zc, pc if b == 64 else truncate(pc, b)) for b in [*bits, 64]} if bits else
                   {"soft": m.decode_soft(zc, pc), "hard": m.decode_static(zc, np.full(n, noise.p_soft)),
                    "informed": m.decode_static(zc, np.full(n, ps.mean()))})
            for k, e in res.items():
                errs[d][k] = errs[d].get(k, 0) + int(e.sum())
        samples[d] = len(z) * (d_full - d + 1)
    return errs, samples
