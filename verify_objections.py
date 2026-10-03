# -*- coding: utf-8 -*-
"""Verify the three decisive reviewer objections. Zero new inference - all from logs on disk.

O1 RELATIVE vs ABSOLUTE. The original reports relative changes ("aptitude drops 16%,
   unreliability increases 112%"). Our margin dU - |dA| uses ABSOLUTE point changes.
   Test: does the RELATIVE version (dU/U_single vs |dA|/A_single) hold in every cell?
   If yes, our absolute margin is testing a claim they never made.

O2 BOUNDED-SUPPORT / FLOOR EFFECT. For a binary scorer, mean U is a flip rate, which is an
   inverted-U function of mean score: zero when all pass, zero when all fail, max in between.
   Test: plot margin against P_multi. If all three tasks lie on ONE curve, the apparent
   task-dependence is just where each task sits on that curve, and is not about task at all.

O3 SCORER CONFOUND. data2text uses BLEU (continuous, compressed range). Binarise it at a
   threshold to match the binary scorers, recompute, and see if its negative margin flips.
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
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in glob.glob(os.path.join(REPO, "logs_run1_frozen", "**", "*.jsonl"), recursive=True):
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            s = r.get("score")
            if s is None:
                continue
            ct = "multi" if str(r.get("conv_type", "")).startswith("sharded") else "single"
            d[(r.get("task"), r.get("assistant_model"))][
                (ct, r.get("task_id") or r.get("instruction_id"))].append(float(s))
    return d


def stats(cells, binarise=None):
    ids = sorted(set(i for (c, i) in cells if c == "single") & set(i for (c, i) in cells if c == "multi"))
    if len(ids) < MIN_INSTR:
        return None
    if min(np.median([len(cells[("single", i)]) for i in ids]),
           np.median([len(cells[("multi", i)]) for i in ids])) < MIN_RUNS:
        return None
    o = {}
    for cond in ("single", "multi"):
        runs = {i: (list(cells[(cond, i)]) if binarise is None
                    else [1.0 if v >= binarise else 0.0 for v in cells[(cond, i)]]) for i in ids}
        o[cond + "_A"] = float(np.mean([np.percentile(runs[i], 90) for i in ids]))
        o[cond + "_U"] = float(np.mean([np.percentile(runs[i], 90) - np.percentile(runs[i], 10) for i in ids]))
        o[cond + "_P"] = float(np.mean([np.mean(runs[i]) for i in ids]))
    o["dA"] = (o["multi_A"] - o["single_A"]) * 100
    o["dU"] = (o["multi_U"] - o["single_U"]) * 100
    o["dP"] = (o["multi_P"] - o["single_P"]) * 100
    o["margin_abs"] = o["dU"] - abs(o["dA"])
    # relative version, as the original reports it
    o["relA"] = (o["multi_A"] / o["single_A"] - 1) * 100 if o["single_A"] > 0 else float("nan")
    o["relU"] = (o["multi_U"] / o["single_U"] - 1) * 100 if o["single_U"] > 0 else float("inf")
    o["margin_rel"] = (o["relU"] - abs(o["relA"])) if np.isfinite(o["relU"]) else float("inf")
    o["n_instr"] = len(ids)
    return o


def main():
    data = load()
    rows = []
    for (t, m), cells in sorted(data.items()):
        s = stats(cells)
        if s:
            rows.append((t, m, s))

    print("=" * 92)
    print("O1: ABSOLUTE vs RELATIVE decomposition")
    print("=" * 92)
    print(f"{'task':10s}{'model':18s}{'dA':>7s}{'dU':>7s}{'absM':>8s} | {'relA%':>8s}{'relU%':>9s}{'rel holds?':>12s}")
    print("-" * 92)
    rel_holds = 0
    for t, m, s in rows:
        rh = (not np.isfinite(s["relU"])) or (s["relU"] > abs(s["relA"]))
        rel_holds += rh
        ru = "inf" if not np.isfinite(s["relU"]) else f"{s['relU']:+.0f}"
        print(f"{t:10s}{m:18s}{s['dA']:>7.1f}{s['dU']:>7.1f}{s['margin_abs']:>8.1f} | "
              f"{s['relA']:>+8.0f}{ru:>9s}{'YES' if rh else 'NO':>12s}")
    print()
    print(f"RELATIVE version holds in {rel_holds}/{len(rows)} cells; "
          f"ABSOLUTE margin positive in {sum(1 for _,_,s in rows if s['margin_abs']>0)}/{len(rows)}")
    if rel_holds == len(rows):
        print(">>> OBJECTION 1 CONFIRMED: the original's RELATIVE claim holds everywhere.")
        print(">>> Our absolute margin therefore tests a DIFFERENT claim than the one they made.")
    else:
        print(">>> Objection 1 NOT fully confirmed: the relative version also fails somewhere.")

    print()
    print("=" * 92)
    print("O2: FLOOR EFFECT - does margin lie on one common curve vs P_multi?")
    print("=" * 92)
    P = np.array([s["multi_P"] * 100 for _, _, s in rows])
    M = np.array([s["margin_abs"] for _, _, s in rows])
    tasks = np.array([t for t, _, _ in rows])
    # fit one global quadratic, then check whether task identity explains the residual
    co = np.polyfit(P, M, 2)
    resid = M - np.polyval(co, P)
    print(f"global quadratic fit R^2 = {1 - resid.var()/M.var():.3f}")
    print("mean residual by task (if ~0 for all, task adds nothing beyond position on the curve):")
    for t in sorted(set(tasks)):
        r = resid[tasks == t]
        print(f"   {t:10s} n={len(r)} mean_resid={r.mean():+7.2f}  sd={r.std():5.2f}")
    between = np.var([resid[tasks == t].mean() for t in sorted(set(tasks))])
    print(f"variance of task-mean residuals = {between:.2f}  "
          f"({'task still matters' if between > 25 else 'task adds little beyond P_multi'})")

    print()
    print("=" * 92)
    print("O3: SCORER CONFOUND - binarise data2text (BLEU) and recompute")
    print("=" * 92)
    for thr in (0.15, 0.25, 0.35):
        out = []
        for (t, m), cells in sorted(data.items()):
            if t != "data2text":
                continue
            s = stats(cells, binarise=thr)
            if s:
                out.append((m, s["margin_abs"]))
        if out:
            vals = [v for _, v in out]
            print(f"  threshold BLEU>={thr}:  margins " +
                  ", ".join(f"{m.split('-')[0]}={v:+.1f}" for m, v in out) +
                  f"   median={np.median(vals):+.1f}  neg={sum(1 for v in vals if v<0)}/{len(vals)}")
    print("  (continuous BLEU baseline: median -8.5, neg 3/4)")

    # ---- figure for O2 ----
    TC = {"math": vs.OKABE_ITO["blue"], "actions": vs.OKABE_ITO["orange"], "data2text": vs.OKABE_ITO["green"]}
    fig, ax = plt.subplots(figsize=(vs.WIDE, 3.1))
    seen = set()
    for t, m, s in rows:
        ax.scatter(s["multi_P"] * 100, s["margin_abs"], s=34, color=TC.get(t, vs.OKABE_ITO["grey"]),
                   edgecolor="black", linewidth=0.5, zorder=3, label=t if t not in seen else None)
        seen.add(t)
    xs = np.linspace(P.min(), P.max(), 100)
    ax.plot(xs, np.polyval(co, xs), ls="--", lw=1.0, color=vs.OKABE_ITO["grey"], zorder=2)
    ax.axhline(0, color="black", lw=0.8, zorder=1)
    ax.set_xlabel(r"multi-turn mean performance $P$ (%)")
    ax.set_ylabel(r"margin: $\Delta U - |\Delta A|$ (pp)")
    vs.legend_above(ax, 3)
    vs.finish(ax)
    vs.save(fig, os.path.join(HERE, "figR5_floor_effect"),
            "Figure R5: Margin against multi-turn mean performance for all model-task cells. "
            "Dashed line is a single quadratic fit ignoring task identity. If the three tasks lay on "
            "one common curve, the apparent task-dependence would reduce to where each task sits on "
            "the bounded score scale rather than to anything about the task.")
    print("\nwrote figR5_floor_effect.pdf/.png")


if __name__ == "__main__":
    main()
