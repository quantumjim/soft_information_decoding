"""Re-decode the hardware data or rerun the simulations (needs data/raw). Writes CSV rows like
data/results.csv.gz to data/rerun/; finished jobs are skipped.

    uv run python scripts/rerun.py hardware --states X0 --rounds 50 --max-jobs 2 --distances 3 5 7
    uv run python scripts/rerun.py hardware --rounds 50 --bits 1 2 3 4 5 6 7 8 9 10 11 15 --distances 3 5 7 9 11 13 15 17 19 21 27
    uv run python scripts/rerun.py sim --state X0 --shots 60000 --hard-p-soft 0           # Fig. 7, as in the paper
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from softinfo import Noise, decode_subsets, postselect, repetition_code
from softinfo.data import load_calibration, load_job, manifest, soft_info

OUT = Path(__file__).parents[1] / "data" / "rerun"
SHERBROOKE = Noise(p2=0.028620063497464567, p1=0.00037101220809731015, t1=0.0013998485571901685,
                   t2=0.004106187996447342, p_hard=0.019907289814579635)  # device noise of the paper's simulations


def write(name, exp, state, T, job, errs, samples, bits):
    with open(OUT / name, "w", newline="") as f:
        csv.writer(f).writerows((exp, state, T, job, d, "soft" if bits else k, k if bits else 64, samples[d], e)
                                for d, by in errs.items() for k, e in by.items())


def hardware(args):
    job, a = args
    name = f"{job['job']}{'_bits' if a.bits else ''}.csv"
    if not (OUT / name).exists():
        iq, qubits, T = load_job(job["job"])
        z, ps = soft_info(iq, qubits, load_calibration(*job["calib"]))
        s = job["state"]
        errs, n = decode_subsets(z, ps, T, s[0] == "X", int(s[1]), Noise.from_list(job["noise_list"]), a.distances, a.bits)
        write(name, "hw" + "_bits" * bool(a.bits), s, T, job["job"], errs, n, a.bits)
    return name


def sim(a):
    """IQ for every simulated outcome: a random post-selected calibration point plus Gaussian KDE jitter (App. F)."""
    rng = np.random.default_rng(a.seed)
    job = next(j for j in manifest()["hardware"] if j["state"] == a.state and j["T"] == a.T)
    _, qubits, _ = load_job(job["job"])
    calib = load_calibration(*job["calib"])
    xbasis, logical = a.state[0] == "X", int(a.state[1])
    noise = Noise(*(a.noise_scale * v for v in vars(SHERBROOKE).values()))
    true = repetition_code(52, a.T, xbasis, logical, noise).compile_sampler(seed=a.seed).sample(a.shots)
    iq = np.empty(true.shape, complex)
    for q in np.unique(qubits):
        cols, clouds = qubits == q, postselect(*calib[q])
        both = np.concatenate(clouds)
        for state, cloud in enumerate(clouds):
            sub, mask = iq[:, cols], true[:, cols] == state
            jitter = rng.normal(0, 0.1, (mask.sum(), 2)) * [both.real.std(), both.imag.std()]
            sub[mask] = cloud[rng.integers(len(cloud), size=mask.sum())] + jitter @ [1, 1j]
            iq[:, cols] = sub
    z, ps = soft_info(iq, qubits, calib)
    noise = replace(noise, p_soft=(z != true).mean() if a.hard_p_soft is None else a.hard_p_soft)
    errs, n = decode_subsets(z, ps, a.T, xbasis, logical, noise, a.distances, a.bits)
    exp = f"sim{a.noise_scale:g}x" + "_bits" * bool(a.bits) + ("" if a.hard_p_soft is None else f"_hardps{a.hard_p_soft:g}")
    write(f"{exp}_{a.state}_{a.T}_{a.shots}_{a.seed}.csv", exp, a.state, a.T, f"seed{a.seed}", errs, n, a.bits)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["hardware", "sim"])
    ap.add_argument("--distances", nargs="+", type=int, default=list(range(3, 52, 2)))
    ap.add_argument("--bits", nargs="*", type=int, default=[])
    ap.add_argument("--states", nargs="+", default=["X0", "X1", "Z0", "Z1"])
    ap.add_argument("--rounds", nargs="+", type=int, default=[10, 20, 30, 40, 50, 75, 100])
    ap.add_argument("--max-jobs", type=int, help="jobs per (state, T)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--state", default="Z0")
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--noise-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hard-p-soft", type=float, help="p_soft of the simulated hard decoder (default: true misassignment "
                                                      "rate; paper: 0 for X0, 0.00517 for Z0)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == "sim":
        sim(a)
    else:
        jobs = [j for s in a.states for T in a.rounds
                for j in [j for j in manifest()["hardware"] if j["state"] == s and j["T"] == T][: a.max_jobs]]
        with ProcessPoolExecutor(a.workers) as ex:
            for i, name in enumerate(ex.map(hardware, [(j, a) for j in jobs])):
                print(f"{i + 1}/{len(jobs)} {name}", flush=True)
