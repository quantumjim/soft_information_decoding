"""Loaders for the exported IBM Sherbrooke data in data/raw (see README)."""
import json
from pathlib import Path

import numpy as np

from .readout import Readout, postselect

RAW = Path(__file__).parents[1] / "data" / "raw"


def unpack(f, key="iq"):
    """Complex IQ from int16 (I, Q) pairs times the stored scale."""
    x = f[key] * f[key + "_scale" if key + "_scale" in f else "scale"]
    return x[..., 0] + 1j * x[..., 1]


def manifest():
    return json.loads((RAW / "manifest.json").read_text())


def load_job(job):
    """(IQ [shots, T*51 + 52], physical qubit of each column, T) of one repetition code job."""
    f = np.load(RAW / "hardware" / f"{job}.npz")
    return unpack(f), f["qubits"], int(f["T"])


def load_calibration(job0, job1):
    """{qubit: (first0, second0, first1, second1)} of a double-measurement calibration pair."""
    a, b = (unpack(np.load(RAW / "calibration" / f"{j}.npz")) for j in (job0, job1))
    n = a.shape[1] // 2
    return {q: (a[:, q], a[:, n + q], b[:, q], b[:, n + q]) for q in range(n)}


def soft_info(iq, qubits, calib):
    """(z_hat, p_soft) of every measurement, each column classified with its qubit's readout model."""
    z, p = np.empty(iq.shape, np.uint8), np.empty(iq.shape)
    for q in np.unique(qubits):
        cols = qubits == q
        z[:, cols], p[:, cols] = Readout(*postselect(*calib[q]))(iq[:, cols])
    return z, p
