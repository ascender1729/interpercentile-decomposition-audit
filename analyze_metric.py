# -*- coding: utf-8 -*-
"""THE PAPER: what the aptitude/unreliability decomposition can and cannot tell you.

Three results, each analytic AND empirical.

R1 NESTED. U_h^l = A_h - P_l by construction, so dU = dA - dP_l and
   margin = dU - |dA| = 2*dA - dP_l when dA <= 0. The "two components" are one
   reparameterisation of (P_h, P_l), not independent axes.

R2 DISAGREEMENT, NOT RELIABILITY. Under a binary scorer with N runs, U_h^l is a deterministic
   function of the success count k, and it is NON-MONOTONE: zero when all runs fail, zero when
   all runs succeed, maximal in between. So it measures whether an instruction is contested,
   not how reliable the model is. Its expectation under Binomial(N, p) is an inverted U in p.

R3 FORCED SIGN AT CEILING. U is a range, so U >= 0. If the single-turn baseline is saturated,
   U_single = 0 and hence dU >= 0 ALWAYS: the metric cannot report a reliability improvement
   even in principle. Since benchmarks are usually chosen so baselines are near ceiling, the
   headline "unreliability increased" is partly guaranteed by construction.

Decisive test: predict each cell's margin from a binomial null using ONLY (P_single, P_multi, N)
and NO task information. If the null reproduces the observed pattern, the apparent
task-dependence is a property of the estimator, not of the tasks.
"""
import json, glob, collections, os, sys
import numpy as np
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "repo")
# viz_style.py ships alongside this file
import viz_style as vs
vs.apply()
import matplotlib.pyplot as plt

MIN_INSTR, MIN_RUNS = 10, 4
BINARY_TASKS = ("math", "actions")


# ---------- R2: analytic behaviour of the estimator under a binary scorer ----------
def u_of_k(k, N, hi=90, lo=10):
    x = [1.0] * k + [0.0] * (N - k)
    x.sort()
    return float(np.percentile(x, hi) - np.percentile(x, lo))


def a_of_k(k, N, hi=90):
    x = [1.0] * k + [0.0] * (N - k)
    x.sort()
    return float(np.percentile(x, hi))


def expected(p, N, fn):
    return sum(comb(N, k) * p ** k * (1 - p) ** (N - k) * fn(k, N) for k in range(N + 1))


def main():
    print("=" * 78)
    print("R2: U is a DISAGREEMENT statistic - non-monotone in the success count")
    print("=" * 78)
    for N in (10, 6, 12):
        print(f"  N={N}:  k -> U90_10 : " +
              "  ".join(f"{k}:{u_of_k(k, N):.2f}" for k in range(N + 1)))
    print("  -> U returns to 0 at BOTH ends. A model that always fails scores as perfectly")
    print("     'reliable', identically to one that always succeeds.")
    print()
    print("  E[U | p, N=6] across the accuracy range:")
    print("   p:    " + "".join(f"{p:>7.1f}" for p in np.arange(0, 1.01, 0.125)))
    print("   E[U]: " + "".join(f"{expected(p, 6, u_of_k):>7.2f}" for p in np.arange(0, 1.01, 0.125)))

    # ---------- load frozen data ----------
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    for p_ in glob.glob(os.path.join(REPO, "logs_run1_frozen", "**", "*.jsonl"), recursive=True):
        for line in open(p_, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("score") is None:
                continue
            ct = "multi" if str(r.get("conv_type", "")).startswith("sharded") else "single"
            data[(r.get("task"), r.get("assistant_model"))][
                (ct, r.get("task_id") or r.get("instruction_id"))].append(float(r["score"]))

    rows = []
    for (t, m), cells in sorted(data.items()):
        ids = sorted(set(i for (c, i) in cells if c == "single") & set(i for (c, i) in cells if c == "multi"))
        if len(ids) < MIN_INSTR:
            continue
        if min(np.median([len(cells[("single", i)]) for i in ids]),
               np.median([len(cells[("multi", i)]) for i in ids])) < MIN_RUNS:
            continue
        o = {}
        for cond in ("single", "multi"):
            o[cond + "_A"] = np.mean([np.percentile(cells[(cond, i)], 90) for i in ids])
            o[cond + "_U"] = np.mean([np.percentile(cells[(cond, i)], 90) - np.percentile(cells[(cond, i)], 10) for i in ids])
            o[cond + "_P"] = np.mean([np.mean(cells[(cond, i)]) for i in ids])
        N = int(np.median([len(cells[("multi", i)]) for i in ids]))
        dA = (o["multi_A"] - o["single_A"]) * 100
        dU = (o["multi_U"] - o["single_U"]) * 100
        margin = dU - abs(dA)
        # binomial null: same accuracies, same N, NO task information
        nA = (expected(o["multi_P"], N, a_of_k) - expected(o["single_P"], N, a_of_k)) * 100
        nU = (expected(o["multi_U"] * 0 + o["multi_P"], N, u_of_k)
              - expected(o["single_P"], N, u_of_k)) * 100
        rows.append(dict(task=t, model=m, N=N, Ps=o["single_P"] * 100, Pm=o["multi_P"] * 100,
                         Us=o["single_U"] * 100, dA=dA, dU=dU, margin=margin,
                         null_margin=nU - abs(nA)))

    print()
    print("=" * 78)
    print("R3: saturated baselines force the sign of dU")
    print("=" * 78)
    sat = [r for r in rows if r["Us"] < 5]
    print(f"  cells with single-turn unreliability U_single < 5pp: {len(sat)} of {len(rows)}")
    print(f"  of those, dU > 0 in {sum(1 for r in sat if r['dU'] > 0)}/{len(sat)} "
          f"(dU >= 0 is FORCED when U_single = 0, since U is a non-negative range)")
    zero = [r for r in rows if r["Us"] == 0.0]
    print(f"  cells with U_single EXACTLY 0 (dU cannot be negative by construction): {len(zero)}")
    for r in zero:
        print(f"     {r['task']}/{r['model']}: dU={r['dU']:+.1f}")

    print()
    print("=" * 78)
    print("DECISIVE TEST: binomial null using only (P_single, P_multi, N), no task information")
    print("=" * 78)
    print(f"{'task':10s}{'model':18s}{'N':>3s}{'observed':>10s}{'null':>8s}{'resid':>8s}")
    for r in sorted(rows, key=lambda x: (x["task"], -x["margin"])):
        print(f"{r['task']:10s}{r['model']:18s}{r['N']:>3d}{r['margin']:>10.1f}"
              f"{r['null_margin']:>8.1f}{r['margin']-r['null_margin']:>8.1f}")
    obs = np.array([r["margin"] for r in rows])
    nul = np.array([r["null_margin"] for r in rows])
    ra = np.argsort(np.argsort(obs)).astype(float)
    rb = np.argsort(np.argsort(nul)).astype(float)
    print()
    print(f"  Pearson  r(observed, null) = {np.corrcoef(obs, nul)[0,1]:+.3f}")
    print(f"  Spearman r(observed, null) = {np.corrcoef(ra, rb)[0,1]:+.3f}")
    print(f"  variance of observed margins explained by the null: "
          f"{max(0.0, 1 - ((obs-nul).var()/obs.var())):.3f}")
    for t in sorted(set(r["task"] for r in rows)):
        s = [r for r in rows if r["task"] == t]
        print(f"    {t:10s} observed median {np.median([r['margin'] for r in s]):+6.1f}   "
              f"null median {np.median([r['null_margin'] for r in s]):+6.1f}")

    # ---------- figure ----------
    TC = {"math": vs.OKABE_ITO["blue"], "actions": vs.OKABE_ITO["orange"], "data2text": vs.OKABE_ITO["green"]}
    fig, axes = plt.subplots(1, 3, figsize=(vs.WIDE, 2.5))

    # panel (a): the estimator's actual shape in the success count
    ax = axes[0]
    ks = list(range(11))
    ax.step(ks, [a_of_k(k, 10) * 100 for k in ks], where="mid", lw=3.4, alpha=0.45,
            solid_joinstyle="miter", color=vs.OKABE_ITO["blue"], label=r"$A_{90}$")
    ax.step(ks, [u_of_k(k, 10) * 100 for k in ks], where="mid", lw=1.3,
            color=vs.OKABE_ITO["orange"], label=r"$U^{10}_{90}$")
    ax.set_ylim(-6, 112)
    ax.set_xlabel("successes $k$ of $N=10$ runs")
    ax.set_ylabel("statistic (pp)")
    ax.set_xticks([0, 2, 5, 8, 10])
    ax.legend(frameon=False, fontsize=7, loc="center right")
    ax.set_title("(a) what the statistics do", fontsize=8)

    ax = axes[1]
    ps = np.linspace(0, 1, 200)
    for N, ls in ((10, "-"), (6, "--")):
        ax.plot(ps * 100, [expected(p, N, u_of_k) * 100 for p in ps], ls,
                color=vs.OKABE_ITO["blue"], lw=1.3, label=f"$N={N}$" + (" (original)" if N==10 else " (ours)"))
    ax.set_xlabel(r"per-instruction accuracy $p$ (%)")
    ax.set_ylabel(r"$\mathbb{E}[U^{10}_{90}]$ (pp)")
    ax.legend(frameon=False, fontsize=7, loc="lower center")
    ax.set_title("(b) expected value vs accuracy", fontsize=8)

    ax = axes[2]
    seen = set()
    for r in rows:
        ax.scatter(r["null_margin"], r["margin"], s=32, color=TC.get(r["task"], vs.OKABE_ITO["grey"]),
                   edgecolor="black", linewidth=0.5, zorder=3,
                   label=r["task"] if r["task"] not in seen else None)
        seen.add(r["task"])
    lim = [min(nul.min(), obs.min()) - 6, max(nul.max(), obs.max()) + 6]
    ax.plot(lim, lim, ls=":", color="black", lw=0.9, zorder=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("binomial-null margin (pp)")
    ax.set_ylabel("observed margin (pp)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title("(c) null vs observed", fontsize=8)
    for a in axes:
        vs.finish(a)
    fig.tight_layout(pad=0.4, w_pad=1.1)
    vs.save(fig, os.path.join(HERE, "figM1_metric"),
            "Figure 1: (a) Under a binary scorer the interpercentile 'unreliability' statistic is a "
            "non-monotone function of accuracy, returning to zero at both extremes: it measures "
            "disagreement among runs, not reliability. (b) Observed per-cell margins against a "
            "binomial null that uses only each cell's single- and multi-turn accuracy and run count, "
            "with no task information. Dotted line is equality.")
    json.dump(rows, open(os.path.join(HERE, "metric_results.json"), "w", encoding="utf-8"), indent=1)
    print("\nwrote figM1_metric.pdf/.png and metric_results.json")


if __name__ == "__main__":
    main()
