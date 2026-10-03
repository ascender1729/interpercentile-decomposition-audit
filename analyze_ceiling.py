# -*- coding: utf-8 -*-
"""CEILING DEGENERACY of the aptitude/unreliability decomposition.

Claim under test: the decomposition's verdict is driven by whether the SINGLE-TURN condition
is saturated, not by multi-turn interaction.

Mechanism (exact, not empirical):
  U = percentile_h(S) - percentile_l(S) is a range, so U >= 0 always.
  If the single-turn condition is saturated, U_single = 0 and A_single = 1. Then
      dU = U_multi - 0   >= 0        (forced non-negative)
      dA = A_multi - 1   <= 0        (forced non-positive)
      margin = dU - |dA| = U_multi + A_multi - 1
  i.e. the margin becomes a function of the MULTI-TURN distribution alone; the single-turn
  condition contributes a constant. The metric cannot report "unreliability decreased" even
  in principle, so the direction of the reported conclusion is baked in by saturation.

Outputs: correlation between single-turn saturation and margin, plus a venue-standard figure.
"""
import json, glob, collections, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "repo")
# viz_style.py ships alongside this file
import viz_style as vs
vs.apply()
import matplotlib.pyplot as plt

MIN_INSTR, MIN_RUNS = 10, 4


def load():
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in glob.glob(os.path.join(REPO, "logs_run1_frozen", "**", "*.jsonl"), recursive=True):
        for line in open(p, encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            s = d.get("score")
            if s is None:
                continue
            ct = "multi" if str(d.get("conv_type", "")).startswith("sharded") else "single"
            data[(d.get("task"), d.get("assistant_model"))][
                (ct, d.get("task_id") or d.get("instruction_id"))].append(float(s))
    return data


def cell_stats(cells, hi=90, lo=10):
    ids = sorted(set(i for (c, i) in cells if c == "single") & set(i for (c, i) in cells if c == "multi"))
    if len(ids) < MIN_INSTR:
        return None
    if min(np.median([len(cells[("single", i)]) for i in ids]),
           np.median([len(cells[("multi", i)]) for i in ids])) < MIN_RUNS:
        return None
    out = {}
    for cond in ("single", "multi"):
        A = np.array([np.percentile(cells[(cond, i)], hi) for i in ids])
        U = np.array([np.percentile(cells[(cond, i)], hi) - np.percentile(cells[(cond, i)], lo) for i in ids])
        P = np.array([np.mean(cells[(cond, i)]) for i in ids])
        out[cond] = (A.mean(), U.mean(), P.mean())
    # saturation: share of instructions whose single-turn runs are ALL perfect
    out["sat"] = float(np.mean([min(cells[("single", i)]) >= 1.0 for i in ids])) * 100
    out["n_instr"] = len(ids)
    out["U_single"] = out["single"][1] * 100
    out["dA"] = (out["multi"][0] - out["single"][0]) * 100
    out["dU"] = (out["multi"][1] - out["single"][1]) * 100
    out["dP"] = (out["multi"][2] - out["single"][2]) * 100
    out["margin"] = out["dU"] - abs(out["dA"])
    return out


def main():
    data = load()
    rows = []
    for (t, m), cells in sorted(data.items()):
        st = cell_stats(cells)
        if st:
            rows.append((t, m, st))

    print(f"{'task':11s}{'model':19s}{'sat%':>6s}{'U_sing':>8s}{'dA':>8s}{'dU':>8s}{'margin':>8s}")
    print("-" * 68)
    for t, m, s in rows:
        print(f"{t:11s}{m:19s}{s['sat']:>5.0f}%{s['U_single']:>8.1f}{s['dA']:>8.1f}{s['dU']:>8.1f}{s['margin']:>8.1f}")

    sat = np.array([s["sat"] for _, _, s in rows])
    us = np.array([s["U_single"] for _, _, s in rows])
    mg = np.array([s["margin"] for _, _, s in rows])

    # Spearman without scipy: correlation of ranks
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    print()
    print(f"cells: {len(rows)}")
    print(f"Pearson  r(single-turn unreliability U_s, margin) = {np.corrcoef(us, mg)[0,1]:+.3f}")
    print(f"Spearman r(U_s, margin)                           = {spearman(us, mg):+.3f}")
    print(f"Pearson  r(saturation, margin)                    = {np.corrcoef(sat, mg)[0,1]:+.3f}")
    print(f"Spearman r(saturation, margin)                    = {spearman(sat, mg):+.3f}")

    forced = [(t, m, s) for t, m, s in rows if s["U_single"] == 0.0]
    print()
    print(f"cells where U_single is EXACTLY 0 (so dU >= 0 is forced, not measured): {len(forced)}")
    for t, m, s in forced:
        print(f"   {t}/{m}: dU={s['dU']:+.1f} (cannot be negative by construction)")
    print(f"cells with margin < 0: {sum(1 for _,_,s in rows if s['margin']<0)}; "
          f"of those, U_single>0 in {sum(1 for _,_,s in rows if s['margin']<0 and s['U_single']>0)}")

    # ---- figure: single-turn unreliability vs margin ----
    TASK_COLOR = {"math": vs.OKABE_ITO["blue"], "actions": vs.OKABE_ITO["orange"],
                  "data2text": vs.OKABE_ITO["green"]}
    fig, ax = plt.subplots(figsize=(vs.WIDE, 3.1))
    seen = set()
    for t, m, s in rows:
        ax.scatter(s["U_single"], s["margin"], s=34, color=TASK_COLOR.get(t, vs.OKABE_ITO["grey"]),
                   edgecolor="black", linewidth=0.5, zorder=3,
                   label=t if t not in seen else None)
        seen.add(t)
    ax.axhline(0, color="black", lw=0.8, zorder=1)
    if len(rows) > 2:
        b, a = np.polyfit(us, mg, 1)
        xs = np.linspace(us.min(), us.max(), 50)
        ax.plot(xs, a + b * xs, ls="--", lw=1.0, color=vs.OKABE_ITO["grey"], zorder=2)
    ax.set_xlabel(r"single-turn unreliability $U^{10}_{90}$ (pp)")
    ax.set_ylabel(r"margin: $\Delta U - |\Delta A|$ (pp)")
    vs.legend_above(ax, 3)
    vs.finish(ax)
    cap = ("Figure R4: The decomposition's verdict tracks how saturated the single-turn condition is. "
           "Each point is one model-task cell. When single-turn unreliability is near zero (left), the "
           "range identity forces dU >= 0 and A_single = 1 forces dA <= 0, so the margin is determined by "
           "the multi-turn distribution alone and the conclusion 'unreliability dominates' cannot fail. "
           "Cells with genuinely variable single-turn behaviour (right, all data-to-text) are where the "
           f"margin goes negative. Spearman rho = {spearman(us, mg):+.2f} over {len(rows)} cells.")
    vs.save(fig, os.path.join(HERE, "figR4_ceiling_degeneracy"), cap)
    print("\nwrote figR4_ceiling_degeneracy.pdf/.png")

    json.dump([{"task": t, "model": m, **{k: v for k, v in s.items() if k not in ("single", "multi")}}
               for t, m, s in rows],
              open(os.path.join(HERE, "ceiling_results.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
