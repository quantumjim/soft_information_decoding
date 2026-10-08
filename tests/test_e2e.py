"""End-to-end tests: each runs a real entry point and checks the artifact it writes."""
import json
import subprocess
import sys
from pathlib import Path

import nbclient
import nbformat
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
needs_raw = pytest.mark.skipif(not (ROOT / "data/raw/manifest.json").exists(), reason="needs data/raw (make data)")
COLS = ["experiment", "state", "T", "job", "d", "method", "bits", "shots", "errors"]


def run(*args):
    return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def test_tutorial_runs_and_soft_beats_hard(tmp_path):
    nb = nbformat.read(ROOT / "notebooks/tutorial.ipynb", 4)
    nb.cells.append(nbformat.v4.new_code_cell(
        'assert rates["soft"] < rates["hard"]\n'
        'assert (sweep["soft"] < sweep["hard"]).all() and by_bits[6] < rates["hard"]'))
    nbclient.NotebookClient(nb, resources={"metadata": {"path": ROOT / "notebooks"}}).execute()
    nbformat.write(nb, tmp_path / "tutorial.ipynb")


def test_figures_reproduce_paper(tmp_path):
    run("scripts/figures.py", tmp_path)
    assert len(list(tmp_path.glob("*.pdf"))) == 18
    table = (tmp_path / "table1.md").read_text()
    for row in ["| |+z⟩ | 1.31 ± 0.01 | 1.63 ± 0.02 | +24.4% |", "| |-z⟩ | 1.36 ± 0.02 | 1.83 ± 0.02 | +34.6% |",
                "| |+x⟩ | 1.36 ± 0.01 | 1.67 ± 0.03 | +22.8% |", "| |-x⟩ | 1.32 ± 0.01 | 1.53 ± 0.02 | +15.9% |",
                "Average increase: +24.4%"]:
        assert row in table


@needs_raw
def test_hardware_rerun_matches_original():
    job = next(j["job"] for j in json.loads((ROOT / "data/raw/manifest.json").read_text())["hardware"] if (j["state"], j["T"]) == ("X0", 10))
    (ROOT / f"data/rerun/{job}.csv").unlink(missing_ok=True)
    run("scripts/rerun.py", "hardware", "--states", "X0", "--rounds", 10, "--max-jobs", 1, "--distances", 3, 5, 7)
    new = pd.read_csv(ROOT / f"data/rerun/{job}.csv", names=COLS).set_index(["d", "method"]).errors
    old = pd.read_csv(ROOT / "data/results.csv.gz").query("job == @job").set_index(["d", "method"]).errors
    ratio = (new / old.reindex(new.index)).unstack()
    assert ratio["hard"].between(0.97, 1.03).all() and ratio["soft"].between(0.95, 1.10).all(), ratio


@needs_raw
def test_simulation_soft_beats_hard():
    out = ROOT / "data/rerun/sim1x_X0_50_300_0.csv"
    out.unlink(missing_ok=True)
    run("scripts/rerun.py", "sim", "--state", "X0", "--shots", 300, "--distances", 5, 9)
    errs = pd.read_csv(out, names=COLS).pivot_table(index="d", columns="method", values="errors")
    assert (errs.soft < errs.hard).all(), errs
