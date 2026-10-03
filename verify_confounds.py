# -*- coding: utf-8 -*-
"""Verify the two objections that could kill the paper.

C1 IDENTITY. U = A90 - P10 by construction, so dU = dA - dP10 and (when dA<=0)
   margin = dU - |dA| = 2*dA - dP10. If exact, the "decomposition" is a reparameterisation
   of (P90, P10), not an independent axis.

C2 RUN-COUNT CONFOUND. math cells were run at N~12, actions cells at N~6. The plug-in
   percentile estimator is biased and its bias depends on N, so the math-vs-actions contrast
   may be an artefact of run count. Test: subsample EVERY cell to a common N=6 and recompute.
"""
import json, glob, collections, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "repo")
RNG = np.random.default_rng(0)
MIN_INSTR = 10


def load():
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    tot = scored = 0
    for p in glob.glob(os.path.join(REPO, "logs_run1_frozen", "**", "*.jsonl"), recursive=True):
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            tot += 1
            s = r.get("score")
            if s is None:
                continue
            scored += 1
            ct = "multi" if str(r.get("conv_type", "")).startswith("sharded") else "single"
            d[(r.get("task"), r.get("assistant_model"))][
                (ct, r.get("task_id") or r.get("instruction_id"))].append(float(s))
    return d, tot, scored


def cell(cells, N=None, reps=1):
    ids = sorted(set(i for (c, i) in cells if c == "single") & set(i for (c, i) in cells if c == "multi"))
    if len(ids) < MIN_INSTR:
        return None
    if N is not None:
        if min(min(len(cells[(c, i)]) for i in ids) for c in ("single", "multi")) < N:
            return None
    acc = []
    for _ in range(reps):
        o = {}
        for cond in ("single", "multi"):
            A, U, P10 = [], [], []
            for i in ids:
                v = cells[(cond, i)]
                if N is not None:
                    v = list(RNG.choice(v, size=N, replace=False))
                A.append(np.percentile(v, 90))
                P10.append(np.percentile(v, 10))
                U.append(np.percentile(v, 90) - np.percentile(v, 10))
            o[cond] = (np.mean(A), np.mean(U), np.mean(P10))
        dA = (o["multi"][0] - o["single"][0]) * 100
        dU = (o["multi"][1] - o["single"][1]) * 100
        dP10 = (o["multi"][2] - o["single"][2]) * 100
        acc.append((dA, dU, dP10, dU - abs(dA)))
    a = np.array(acc)
    return dict(dA=a[:, 0].mean(), dU=a[:, 1].mean(), dP10=a[:, 2].mean(),
                margin=a[:, 3].mean(), margin_sd=a[:, 3].std(), n_instr=len(ids),
                minruns=min(min(len(cells[(c, i)]) for i in ids) for c in ("single", "multi")))


data, tot, scored = load()
print(f"log records: {tot} total, {scored} scored, {tot-scored} unscored")
print()

rows = [(t, m, cell(c)) for (t, m), c in sorted(data.items())]
rows = [(t, m, s) for t, m, s in rows if s]

print("=" * 78)
print("C1: is margin == 2*dA - dP10 exactly? (identity check)")
print("=" * 78)
print(f"{'task':10s}{'model':18s}{'margin':>9s}{'2dA-dP10':>10s}{'diff':>8s}")
mx = 0.0
for t, m, s in rows:
    pred = 2 * s["dA"] - s["dP10"]
    d = abs(pred - s["margin"])
    mx = max(mx, d)
    print(f"{t:10s}{m:18s}{s['margin']:>9.2f}{pred:>10.2f}{d:>8.4f}")
print()
print(f"max |difference| = {mx:.6f}  -> "
      f"{'IDENTITY CONFIRMED: margin is a reparameterisation of (P90,P10)' if mx < 0.01 else 'not an identity'}")

print()
print("=" * 78)
print("C2: run-count confound - runs per instruction by task")
print("=" * 78)
for t in sorted(set(r[0] for r in rows)):
    mins = [s["minruns"] for tt, _, s in rows if tt == t]
    print(f"  {t:10s} min runs/instruction per cell: {mins}")

print()
print("=" * 78)
print("C2: SUBSAMPLE EVERY CELL TO COMMON N=6 (20 resamples) - does the gap survive?")
print("=" * 78)
print(f"{'task':10s}{'model':18s}{'orig':>8s}{'N=6':>9s}{'sd':>6s}")
out = collections.defaultdict(list)
orig = collections.defaultdict(list)
for t, m, s in rows:
    s6 = cell(data[(t, m)], N=6, reps=20)
    if s6:
        out[t].append(s6["margin"])
        orig[t].append(s["margin"])
        print(f"{t:10s}{m:18s}{s['margin']:>8.1f}{s6['margin']:>9.1f}{s6['margin_sd']:>6.1f}")
    else:
        print(f"{t:10s}{m:18s}{s['margin']:>8.1f}{'--':>9s}{'':>6s}  (fewer than 6 runs)")

print()
print(f"{'task':10s}{'n':>3s}{'orig median':>13s}{'N=6 median':>12s}")
for t in sorted(out):
    print(f"{t:10s}{len(out[t]):>3d}{np.median(orig[t]):>13.1f}{np.median(out[t]):>12.1f}")
if len(out) >= 2 and "math" in out and "actions" in out:
    g0 = np.median(orig["math"]) - np.median(orig["actions"])
    g1 = np.median(out["math"]) - np.median(out["actions"])
    print()
    print(f"math-minus-actions gap: original {g0:+.1f} pp  ->  at common N=6 {g1:+.1f} pp")
    print(">>> " + ("GAP SURVIVES the run-count control" if g1 > 10 else
                   "GAP SHRINKS SUBSTANTIALLY - run count explains much of it"))
