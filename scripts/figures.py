"""All data figures of the paper and Table I, from data/results.csv.gz and data/iq_figures.npz.

    uv run python scripts/figures.py [out_dir]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from softinfo import Readout, postselect, truncate
from softinfo.data import unpack

DATA = Path(__file__).parents[1] / "data"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parents[1] / "figures")
W = 140 / 25.4
H = W * 2 / (1 + 5**0.5)
STATES = {"Z0": "+z", "Z1": "-z", "X0": "+x", "X1": "-x"}
LABEL = {"soft": "Soft MWPM", "hard": "Hard MWPM", "informed": "Data-informed Hard MWPM"}
COLOR = {"soft": ("skyblue", "lightcoral"), "hard": ("midnightblue", "darkred"), "informed": ("steelblue", "tomato")}
MARKER = {"0": "o", "1": "d"}
MIN_ERRS = 5
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10, "legend.fontsize": 8,
                     "figure.figsize": (W, H), "axes.formatter.use_mathtext": True, "xtick.direction": "in",
                     "ytick.direction": "in", "axes.axisbelow": True})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", dpi=300)
    plt.close(fig)


def per_round(PL, T):
    """Logical error per round from the total after T rounds (Eq. 10)."""
    return 0.5 * (1 - (1 - 2 * PL) ** (1 / T))


def fit_lambda(d, eps):
    """Λ, its std and the intercept of log10 ε = c - (d+1)/2 log10 Λ (Eq. 8)."""
    (slope, c), cov = np.polyfit((d + 1) / 2, np.log10(eps), 1, cov=True)
    return 10**-slope, 10**-slope * np.log(10) * cov[0, 0] ** 0.5, c


def wilson(p, n, z=1.0):
    mid, half = (p + z**2 / (2 * n)) / (1 + z**2 / n), z * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5 / (1 + z**2 / n)
    return mid - half, mid + half


counts = pd.read_csv(DATA / "results.csv.gz").groupby(["experiment", "state", "T", "method", "bits", "d"])[["shots", "errors"]].sum()


def curve(exp, state, T, method, bits=64):
    x = counts.loc[(exp, state, T, method, bits)]
    return x.index.values, x.errors.values, x.shots.values


def lam(exp, state, T, method):
    d, k, n = curve(exp, state, T, method)
    return fit_lambda(d[k > MIN_ERRS], per_round(k / n, T)[k > MIN_ERRS])


def eps_plot(ax, exp, state, T, method, color, labels):
    """Logical error per round vs d with Wilson band, outliers (≤ 5 errors) dotted and the Λ fit."""
    d, k, n = curve(exp, state, T, method)
    eps, ok, m = per_round(k / n, T), k > MIN_ERRS, MARKER[state[1]]
    lo, hi = wilson(eps, n * T)
    tail = np.r_[np.flatnonzero(ok)[-1], np.flatnonzero(~ok)]
    for sel, style in ((ok, dict(marker=m, ms=2, lw=1)), (tail, dict(ls=":", lw=0.6))):
        ax.fill_between(d[sel], lo[sel], hi[sel], color="k", alpha=0.1, lw=0)
        ax.plot(d[sel], eps[sel], color=color, **style)
    ax.plot(d[~ok], eps[~ok], m, color=color, ms=3, mfc="none", mew=0.5)
    L, err, c = fit_lambda(d[ok], eps[ok])
    ax.plot(d, 10 ** (c - (d + 1) / 2 * np.log10(L)), "--", color=color, lw=0.4)
    names = labels(L, err)
    return [Line2D([], [], color=color, marker=m, ms=2, lw=1, label=names[0])] + \
        [Line2D([], [], color=color, ls="--", lw=0.4, label=x) for x in names[1:]]


def finish(ax, handles, title, extras=True):
    if extras:
        handles += [Patch(color="k", alpha=0.1, label="68% CI (Wilson)"),
                    Line2D([], [], color="grey", ls=":", marker="o", ms=3, mfc="none", mew=0.5, lw=0.6,
                           label=rf"Outliers $n_{{errs}} \leq {MIN_ERRS}$")]
    ax.legend(handles=handles, loc="upper right")
    ax.set(yscale="log", ylim=(1e-8, 2e-2), xlabel="Distance", ylabel="Logical error per round", xticks=range(3, 52, 8), title=title)
    ax.grid(True, which="both", ls="--", lw=0.2)


def fig_lambda(exp, state, methods, name):
    fig, ax = plt.subplots()
    hs = [h for m in methods for h in eps_plot(ax, exp, state, 50, m, COLOR[m][state[0] == "X"],
                                               lambda L, e, m=m: (LABEL[m], rf"$\Lambda$-Fit: $\Lambda$={L:.2f}$\pm${e:.2f}"))]
    finish(ax, hs, rf"Prepared logical state $|{STATES[state]}\rangle_L$")
    save(fig, name)


def fig_rounds(method, rounds=(100, 75, 50, 40, 30, 20, 10)):
    fig, ax = plt.subplots()
    colors = plt.get_cmap("viridis" if method == "soft" else "inferno")(np.linspace(0, 0.9, len(rounds)))
    hs = [h for T, c in zip(rounds, colors) for h in eps_plot(ax, "hw", "X0", T, method, c, lambda L, e, T=T: (rf"T = {T}, $\Lambda$={L:.2f}$\pm${e:.2f}",))]
    finish(ax, hs, LABEL[method], extras=False)
    save(fig, f"fig12_{method}_rounds.pdf")


def fig_bits(exp, panels, name):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(W, W * 0.75))
    for ax, (state, max_d) in zip(axs, panels):
        ds = [d for d in counts.loc[(exp, state)].index.unique("d") if d <= max_d]
        for d, c in zip(ds, plt.get_cmap("Reds" if state[0] == "X" else "Blues")(np.linspace(0.2, 1, len(ds)))):
            x = counts.loc[(exp, state, 50, "soft")].xs(d, level="d").errors
            ax.plot(x.index[:-1], x.values[:-1] / x[64], marker=MARKER[state[1]], color=c, ms=2, lw=1, label=f"d={d}")
        ax.axvline(8, color="k", ls="--", lw=1, alpha=0.6)
        ax.text(8.15, np.mean(ax.get_ylim()), "1 byte", rotation=90, va="center", fontsize=8)
        ax.grid(True, which="both", ls="--", lw=0.2)
        ax.legend(ncol=2, fontsize=7)
    axs[1].set_xlabel(r"Number of bits $b$ used for $p^b_s(\mu)$")
    fig.supylabel(r"Logical error probability ratio $P^b_L / P^{64}_L$", fontsize=10)
    save(fig, name)


def gains(method, T):
    """Λ_method / Λ_hard - 1 per state and its error (propagated)."""
    out = []
    for s in STATES:
        (Lm, em, _), (Lh, eh, _) = lam("hw", s, T, method), lam("hw", s, T, "hard")
        out.append((Lm / Lh - 1, np.hypot(em / Lh, Lm * eh / Lh**2)))
    return np.array(out)


def fig_threshold_vs_rounds(rounds=(10, 20, 30, 40, 50, 75, 100)):
    fig, ax = plt.subplots()
    for method, color in [("soft", "forestgreen"), ("informed", "chocolate")]:
        g = [gains(method, T) for T in rounds]
        mean = 100 * np.array([x[:, 0].mean() for x in g])
        ax.errorbar(rounds, mean, yerr=[100 * np.sqrt((x[:, 1] ** 2).sum()) / len(x) for x in g], fmt="o-", color=color, ms=5, lw=2, label=LABEL[method])
    xs = np.linspace(10, 101, 100)
    ys = np.interp(xs, rounds, 100 * np.array([gains("soft", T)[:, 0].mean() for T in rounds]))
    ax.fill_between(xs[ys >= 20], 20, ys[ys >= 20], color="limegreen", alpha=0.3, lw=0)
    ax.axhline(20, color="k", ls="--", lw=1, alpha=0.5)
    ax.text(75, 20.5, r"$\geq$20% higher threshold", fontsize=8, color="grey")
    ax.set(xlabel="Rounds", ylabel="Average threshold improvement (%)", ylim=(-3, 28), xlim=(5, 105), yticks=range(0, 26, 5))
    ax.grid(which="both", ls="--", lw=0.5)
    ax.legend()
    save(fig, "fig13_threshold_vs_rounds.pdf")


def table1():
    rows, incr = ["| State | Hard Λ | Soft Λ | Increase |", "|---|---|---|---|"], []
    for s in STATES:
        (h, he, _), (so, se, _) = lam("hw", s, 50, "hard"), lam("hw", s, 50, "soft")
        incr.append(round(so, 2) / round(h, 2) - 1)  # from the printed values, as in the paper
        rows.append(f"| |{STATES[s]}⟩ | {h:.2f} ± {he:.2f} | {so:.2f} ± {se:.2f} | +{100 * incr[-1]:.1f}% |")
    (OUT / "table1.md").write_text("\n".join(rows) + f"\n\nAverage increase: +{100 * np.mean(incr):.1f}%\n")


IQ = np.load(DATA / "iq_figures.npz")
IQ = {k: unpack(IQ, k) for k in IQ.files if k + "_scale" in IQ.files}


def readout(q):
    return Readout(*postselect(*(IQ[f"cal_q{q}_{k}"] for k in ("first0", "second0", "first1", "second1"))))


def scatter_hist(clouds, colors, hist_colors, name, labels=(), texts=(), figsize=(W, W * 0.62), right=None):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 8 if right else 4, hspace=0.05, wspace=0.05)
    ax = fig.add_subplot(gs[1:, :3])
    top, side = fig.add_subplot(gs[0, :3], sharex=ax), fig.add_subplot(gs[1:, 3], sharey=ax)
    for c, col, (hx, hy) in zip(clouds, colors, hist_colors):
        ax.scatter(c.real, c.imag, s=0.1, marker=".", color=col, alpha=min(1, max(2e4 / len(c), 2e-3)), rasterized=True)
        top.hist(c.real, bins=250, color=hx, alpha=0.3)
        side.hist(c.imag, bins=250, color=hy, alpha=0.3, orientation="horizontal")
    for lab, col in zip(labels, colors):
        ax.scatter([], [], s=8, color=col, label=lab)
    if labels:
        ax.legend(loc="upper left")
    for lab, z in texts:
        ax.text(np.median(z.real), np.median(z.imag), lab, ha="center", va="center", fontsize=12)
    ax.set(xlabel="In-Phase [arb.]", ylabel="Quadrature [arb.]")
    top.set_ylabel("Counts")
    side.set_xlabel("Counts")
    top.tick_params(labelbottom=False)
    side.tick_params(labelleft=False)
    if right:
        right(fig, gs, top)
    save(fig, name)


def fig15_surface(fig, gs, top):
    c = IQ["fig15_q106_1"]
    h, xe, ye = np.histogram2d(c.real, c.imag, bins=150)
    x, y = np.meshgrid((xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2, indexing="ij")
    ax = fig.add_subplot(gs[:, 4:], projection="3d")
    ax.plot_surface(x, y, h, cmap="plasma")
    ax.set(xlabel="In-Phase [arb.]", ylabel="Quadrature [arb.]", zlabel="Counts", title="3D Histogram Heatmap", xticklabels=[], yticklabels=[])
    top.set_title("Scatter Plot")


def fig17(bits=2):
    pts = IQ["exp_q72"][:1000].ravel()
    _, p = readout(72)(pts, leakage=False)  # the paper shows p_s without the leakage override
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(W, H * 1.5))
    for ax, pp, b in [(axs[0], p, 64), (axs[1], truncate(p, bits), bits)]:
        sc = ax.scatter(pts.real, pts.imag, c=pp, cmap="viridis", vmin=0, vmax=0.5, s=0.1, alpha=0.7, rasterized=True)
        ax.set(title=rf"{b}-bit-accuracy $p^{{{b}}}_s(\mu)$", ylabel="Quadrature [arb.]")
    axs[1].set_xlabel("In-Phase [arb.]")
    fig.colorbar(sc, ax=axs, label=r"Soft flip probability $p_s(\mu)$")
    save(fig, "fig17_ps_q72.pdf")


def fig18(bits=5):
    _, p = readout(72)(IQ["exp_q72"].ravel(), leakage=False)
    fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[1, 2], hspace=0.1))
    for ax in axs:
        ax.hist(truncate(p, bits), bins=150, color="k", alpha=0.6, label=rf"{bits}-bit-accuracy $p^{bits}_s(\mu)$")
        ax.hist(p, bins=150, color="goldenrod", alpha=0.7, label=r"64-bit-accuracy $p^{64}_s(\mu)$")
        ax.legend()
    axs[0].set_ylabel("Frequency")
    axs[1].set(yscale="log", ylabel="Frequency (log scale)", xlabel="Soft flip probability")
    save(fig, "fig18_ps_hist_q72.pdf")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    scatter_hist([IQ["fig2_q72_0"], IQ["fig2_q72_1"]], ["steelblue", "purple"], [("steelblue",) * 2, ("purple",) * 2],
                 "fig02_iq_q72.pdf", labels=(r"$|0\rangle$ state", r"$|1\rangle$ state"))
    q3 = IQ["exp_q3"].ravel()
    scatter_hist([q3[readout(3).leaked(q3)]], ["red"], [("k", "k")], "fig06_outliers_q3.pdf",
                 texts=[(r"$|0\rangle$", IQ["cal_q3_first0"]), (r"$|1\rangle$", IQ["cal_q3_first1"])])
    for s in ("Z0", "X0"):
        fig_lambda("sim1x", s, ["soft", "hard"], f"fig07_sim_{s}.pdf")
    for s in STATES:
        fig_lambda("hw", s, ["soft", "hard"], f"fig08_hw_{s}.pdf")
    for s in ("Z0", "X1"):
        fig_lambda("hw", s, ["soft", "hard", "informed"], f"fig09_informed_{s}.pdf")
    fig_bits("sim2x_bits", [("X0", 33), ("Z0", 27)], "fig10_sim_bits.pdf")
    fig_bits("hw_bits", [("X0", 27), ("Z1", 27)], "fig11_hw_bits.pdf")
    for m in ("soft", "hard"):
        fig_rounds(m)
    fig_threshold_vs_rounds()
    scatter_hist([IQ["fig15_q106_1"]], ["steelblue"], [("blue", "red")], "fig15_iq_q106.pdf", figsize=(W * 1.3, H), right=fig15_surface)
    fig17()
    fig18()
    table1()
    print(f"wrote {len(list(OUT.iterdir()))} files to {OUT}")
