"""One-off provenance: how data/ was made from the original thesis repository and the IBM archive.

    uv run python -m softinfo.export results                       # decoded counts -> data/results.csv.gz
    uv run python -m softinfo.export raw "Soft Info Data.zip"      # jobs used in the paper -> data/raw (int16 IQ)
    uv run python -m softinfo.export figures                       # IQ for Figs. 2, 6, 15, 17, 18 -> data/iq_figures.npz
"""
import csv, gzip, json, re, sys, zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

OLD = Path("~/projects/others/soft_information_decoding").expanduser()
RESULTS = OLD / "notebooks/Decoding/_Thesis/_Thesis_decoding/results"
DATA = Path(__file__).parents[1] / "data"
METHODS = {"s_KPS": "soft", "h_KPS": "hard", "h_K_meanPS": "informed", "s_KPS_no_ambig": "soft_noleak", "s_G": "soft_gauss", "h_G": "hard_gauss"}
HW = [(s, T) for s in ("X0", "X1", "Z0", "Z1") for T in (10, 20, 30, 40, 50, 75, 100)]


def results():
    files = [(f"ibm_sherbrooke_{s}_{T}", "hw", s, T) for s, T in HW]
    files += [(f"simulations/{m}_ibm_sherbrooke_{s}_50", f"sim{m}x", s, 50) for m in (1, 2) for s in ("X0", "Z0")]
    files += [(f"InfoPerfo/ibm_sherbrooke_{s}_50", "hw_bits", s, 50) for s in ("X0", "X1", "Z0", "Z1")]
    files += [(f"simulations/InfoPerfo/{m}_ibm_sherbrooke_{s}_50", f"sim{m}x_bits", s, 50) for m in (2, 3) for s in ("X0", "Z0")]
    rows = []
    for name, exp, state, T in files:
        for job, c in json.load(open(RESULTS / f"{name}.json")).items():
            bits = c["additional_info"].get("nBits_list")
            for d, entry in c["distances"].items():
                n = int(entry["tot_shots_with_all_subsets"])
                if bits:
                    rows += [(exp, state, T, job, int(d), "soft", 64 if b == -1 else b, n, e) for b, e in zip(bits, entry["dict_s_KPS"]["errs_per_bit"])]
                else:
                    rows += [(exp, state, T, job, int(d), METHODS[k], 64, n, int(v["sum_errs"])) for k, v in entry.items() if k in METHODS]
    with gzip.open(DATA / "results.csv.gz", "wt", newline="") as f:
        csv.writer(f).writerows([("experiment", "state", "T", "job", "d", "method", "bits", "shots", "errors"), *sorted(rows)])


def plan():
    """Jobs of the paper with the calibration pair the original pipeline picked (closest by execution date)."""
    meta = {m["job_id"]: m for m in json.load(open(OLD / ".Scratch/job_metadata.json"))}
    cal = pd.DataFrame([{**m["additional_metadata"], "job_id": j, "creation_date": m["creation_date"]} for j, m in meta.items()
                        if m["backend_name"] == "ibm_sherbrooke" and "sampled_state" in (m["additional_metadata"] or {})])
    cal = cal[(cal.job_status == "JobStatus.DONE") & (cal.optimization_level == 0)]
    cal = cal.assign(execution_date=pd.to_datetime(cal.execution_date, utc=True, format="ISO8601"), state=cal.sampled_state.str[0],
                     creation_date=pd.to_datetime(cal.creation_date, utc=True, format="ISO8601"), double=cal.double_msmt.fillna(False).astype(bool))

    def closest(date, double):
        sub = [cal[(cal.double == double) & (cal.state == s)] for s in "01"]
        pick = [x.iloc[(x.execution_date - date).abs().argsort()[:1]] for x in sub]
        return [p.job_id.values[0] for p in pick], pick[0].creation_date.values[0]

    jobs = []
    for s, T in HW:
        for job, c in json.load(open(RESULTS / f"ibm_sherbrooke_{s}_{T}.json")).items():
            date = pd.to_datetime(meta[job]["additional_metadata"]["execution_date"], utc=True)
            pair, _ = closest(pd.to_datetime(closest(date, False)[1], utc=True), True)
            jobs.append(dict(job=job, state=s, T=T, calib=pair, noise_list=c["additional_info"]["noise_list"],
                             p_soft_data=c["additional_info"].get("pSoft_PS_mean"), execution_date=str(date)))
    return jobs


def export_job(args):
    """One gzipped-JSON job -> npz with int16 IQ and a float scale per column (error ~2e-4 of the cloud width)."""
    zip_path, member, out, layout = args
    with zipfile.ZipFile(zip_path) as zf:
        d = json.loads(gzip.decompress(zf.read(member)))["__value__"]
    mem = np.asarray(d["result"]["__value__"]["results"][0]["data"]["memory"], dtype=np.int64)
    scale = np.where(np.abs(mem).max(0) > 0, np.abs(mem).max(0), 1) / 32767
    extra = {}
    if layout:
        regs = {}
        for k, v in d["initial_layouts"][0].items():
            name, i = re.search(r"'(\w+)'\), (\d+)", k).groups()
            regs.setdefault(name, {})[int(i)] = int(v)
        link, code = ([regs[r][i] for i in sorted(regs[r])] for r in ("link_qubit", "code_qubit"))
        extra = dict(qubits=np.array(link * layout["T"] + code, np.int16), T=layout["T"], state=layout["state"])
    np.savez(out, iq=np.round(mem / scale).astype(np.int16), scale=scale.astype(np.float32), **extra)
    return out.name


def raw(zip_path):
    jobs = plan()
    calibs = sorted({c for j in jobs for c in j["calib"]}) + ["cqz92xkczq6g0081h1sg", "cqz92z3dvs8g008j5y90"]  # + Fig. 15
    with zipfile.ZipFile(zip_path) as zf:
        members = {re.search(r"-(\w+)\.json\.gz$", n).group(1): n for n in zf.namelist() if n.endswith(".json.gz")}
    for sub in ("hardware", "calibration"):
        (DATA / "raw" / sub).mkdir(parents=True, exist_ok=True)
    (DATA / "raw" / "manifest.json").write_text(json.dumps({"hardware": jobs, "calibration": calibs}, indent=1))
    tasks = [(zip_path, members[c], DATA / "raw/calibration" / f"{c}.npz", None) for c in calibs]
    tasks += [(zip_path, members[j["job"]], DATA / "raw/hardware" / f"{j['job']}.npz", j) for j in jobs]
    with ProcessPoolExecutor(8) as ex:
        for i, name in enumerate(ex.map(export_job, [t for t in tasks if not t[2].exists()])):
            print(i + 1, name, flush=True)


def figures():
    from softinfo.data import load_calibration, load_job, manifest, unpack
    cal = load_calibration("cqz92ybczq6g0081h1t0", "cqz92zvdvs8g008j5y9g")
    arrays = dict(fig2_q72_0=cal[72][0], fig2_q72_1=cal[72][2],
                  fig15_q106_1=unpack(np.load(DATA / "raw/calibration/cqz92z3dvs8g008j5y90.npz"))[:, 106])
    jobs = [j for j in manifest()["hardware"] if j["state"] == "Z0" and j["T"] == 100]
    pair = Counter(tuple(j["calib"]) for j in jobs).most_common(1)[0][0]
    group = [j["job"] for j in jobs if tuple(j["calib"]) == pair]
    cal = load_calibration(*pair)
    for q in (3, 72):
        arrays |= {f"cal_q{q}_{k}": a for k, a in zip(("first0", "second0", "first1", "second1"), cal[q])}
        arrays[f"exp_q{q}"] = np.vstack([iq[:, qs == q] for iq, qs, _ in map(load_job, group)])
    arrays["exp_q72"] = arrays["exp_q72"][:2000]
    packed = {}
    for k, v in arrays.items():
        packed[k + "_scale"] = max(np.abs(v.real).max(), np.abs(v.imag).max()) / 32767
        packed[k] = np.round(np.stack([v.real, v.imag], -1) / packed[k + "_scale"]).astype(np.int16)
    np.savez_compressed(DATA / "iq_figures.npz", **packed, calib_pair=np.array(pair), jobs=np.array(group))


if __name__ == "__main__":
    {"results": results, "raw": lambda: raw(sys.argv[2]), "figures": figures}[sys.argv[1]]()
