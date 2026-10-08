# Soft information decoding

Soft information decoding of repetition codes on superconducting hardware, as in
M. D. Hanisch, B. Hetényi, J. R. Wootton, *Soft information decoding with superconducting qubits*,
[APS Open Science (2026)](https://doi.org/10.1103/y9fh-4x6n), [arXiv:2411.16228](https://arxiv.org/abs/2411.16228).

This is a cleaned up and simplified version of the code behind the paper, in pure Python on stim and
PyMatching. The original C++ implementation is in the git history (commit `fba86b4`).

## Getting started

Needs [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/quantumjim/soft_information_decoding.git
cd soft_information_decoding
make setup
```

Then open `notebooks/tutorial.ipynb` with the `.venv` kernel and run it. It walks through the readout
model, leakage, the circuit, soft vs hard decoding and the hardware results, using only data shipped
with the repo.

<details>
<summary><b>Additional info</b></summary>

### Make targets

```bash
make figures   # every data figure of the paper and Table I -> figures/
make test      # end-to-end tests (data-dependent ones are skipped without data/raw)
make data RAW_TAR=path/to/softinfo-raw-v1.tar   # unpack the raw IQ data -> data/raw
make hardware  # decode one real IBM job (needs data/raw)
make sim       # small simulation of the device (needs data/raw)
make rerun-all # re-decode all hardware jobs
```

### Files

| File | Paper |
|---|---|
| `softinfo/circuit.py` | repetition code without ancilla resets, circuit-level noise (Fig. 1, App. C) |
| `softinfo/readout.py` | KDE of the IQ clouds on a grid, soft flip probability, leakage → p_s = 0.5, calibration post-selection (Eqs. 12, 17, Sec. II F, App. D) |
| `softinfo/decode.py` | per-shot reweighted matching (Algorithm 1, Eqs. 18, B1), static hard decoders, b-bit truncation, subsampling (App. E, G) |
| `softinfo/data.py` | loaders for `data/raw` |
| `softinfo/export.py` | how `data/` was produced from the original results and the IBM archive |
| `notebooks/tutorial.ipynb`, `scripts/figures.py`, `scripts/rerun.py` | walkthrough, figures, re-decoding of the hardware data and simulations |

### Data

| File | Size | Content |
|---|---|---|
| `data/results.csv.gz` | 0.3 MB | logical errors per job, distance and decoder of the original pipeline (Figs. 7 to 13, Table I) |
| `data/iq_figures.npz` | 7.5 MB | IQ points of qubits 3, 72, 106 (Figs. 2, 6, 15, 17, 18) |
| `data/raw/` | 8.5 GB | IQ of all 554 IBM Sherbrooke jobs and 28 calibrations used in the paper, not in git: `make data RAW_TAR=softinfo-raw-v1.tar` |

`results.csv.gz` columns: experiment (`hw`, `hw_bits`, `sim1x`, `sim2x_bits`, ...), state (`Z0` = |+z⟩,
`Z1` = |−z⟩, `X0` = |+x⟩, `X1` = |−x⟩), rounds T, job, distance d, method (`soft`, `hard`, `informed`,
and for T = 50 `soft_noleak`, `soft_gauss`, `hard_gauss`), bits of p_s (64 = full), decoded samples
(shots × sub-chains), logical errors. `data/raw/hardware/<job>.npz` holds int16 IQ `[shots, T·51 + 52, 2]`
with a per-column scale (error ≈ 2·10⁻⁴ of the cloud width) and the physical qubit of each column;
`manifest.json` lists each job's state, T, calibration pair and device noise. The 132 GB IBM archive
holds 111 GB of jobs the paper does not use; `softinfo/export.py` documents how `data/` was produced.

### Rerunning

`make hardware` and `make sim` decode small subsets; `scripts/rerun.py --help` lists all options (states,
rounds, distances, bit truncation, simulated hard decoder). Results go to `data/rerun/` in the format of
`results.csv.gz`. A full rerun is about 340 CPU hours. Compared with the original C++ code, the hard
decoder reproduces the original error counts within about 1 %, the soft decoder gives about 9 % more
errors (the original counted the mean p_s twice on last-round edges and used an approximate KDE).

</details>

## Differences from the paper text

**Readout.** The readout model is the one behind the published numbers, which differs slightly from the
paper text: a fixed KDE bandwidth of 0.6 instead of a cross-validated one, and a point counts as leaked
when both KDE densities are below 0.01 (z-scored IQ units) instead of the 1 % sampling-probability rule.

**Simulated hard decoder.** In the simulations of Fig. 7 the hard decoder for |+x⟩ assumed no soft
flips at all (p_s = 0), although the simulated readout misassigns about 0.5 % of the outcomes; the |+z⟩
run used 0.0052. A hard decoder that ignores readout errors is a weaker baseline than the one used on
hardware, which always had the calibrated value, so the simulated soft gain was inflated. Here the
simulated hard decoder uses the true misassignment rate by default (`scripts/rerun.py sim --hard-p-soft 0`
gives the paper's baseline). This closes the gap between simulation and hardware. For |+x⟩, T = 50,
d = 9 to 13:

| | Λ hard | Λ soft | Soft gain |
|---|---|---|---|
| Hardware | 1.35 | 1.51 | +12 % |
| Simulation, hard decoder with p_s = 0 (paper) | 2.22 | 2.75 | +24 % |
| Simulation, hard decoder with calibrated p_s | 2.40 | 2.75 | +15 % |

## Citation

```bibtex
@article{hanisch2026soft,
  title   = {Soft information decoding with superconducting qubits},
  author  = {Hanisch, Maurice D. and Het{\'e}nyi, Bence and Wootton, James R.},
  journal = {APS Open Science},
  year    = {2026},
  doi     = {10.1103/y9fh-4x6n},
  eprint  = {2411.16228},
  archivePrefix = {arXiv}
}
```
