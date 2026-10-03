# -*- coding: utf-8 -*-
"""THE OPERATIONAL TEST: is the multi-turn loss actually recoverable by retrying?

The decomposition's practical promise is that capability is preserved and only consistency is
lost. That has a directly testable consequence: if the model is still able to solve the
instruction and merely does so inconsistently, then RETRYING should recover the loss. If
best-of-K over multi-turn runs does NOT close the gap to single-turn, then the capability is
genuinely gone and the "unreliable but capable" reading has no operational content.

Metrics per (task, model) cell, using only instructions paired across both conditions:
  single_any    = fraction of instructions solved by at least one single-turn run
  multi_any     = fraction solved by at least one multi-turn run  (ORACLE best-of-K)
  multi_mean    = ordinary mean multi-turn score
  recovered     = (multi_any - multi_mean) / (single_mean - multi_mean)   share of gap retries close
  unrecoverable = single_any - multi_any                                  gap oracle retry cannot close
  deterministic = fraction of instructions solved in EVERY single-turn run and NO multi-turn run

Bootstrap over instructions for the unrecoverable gap.
"""
import json, glob, collections, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "repo")
RNG = np.random.default_rng(0)
MIN_INSTR, MIN_RUNS = 10, 4
THR = 0.999  # a run counts as a success at full credit (binary scorers); BLEU handled separately


def load():
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in glob.glob(os.path.join(REPO, "logs_run1_frozen", "**", "*.jsonl"), recursive=True):
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("score") is None:
                continue
            ct = "multi" if str(r.get("conv_type", "")).startswith("sharded") else "single"
            d[(r.get("task"), r.get("assistant_model"))][
                (ct, r.get("task_id") or r.get("instruction_id"))].append(float(r["score"]))
    return d


def main():
    data = load()
    rows = []
    print(f"{'task':10s}{'model':18s}{'sing':>7s}{'multi':>7s}{'best-K':>8s}"
          f"{'recov%':>8s}{'unrec':>7s}{'unrec CI':>16s}{'det%':>6s}")
    print("-" * 90)
    for (t, m), cells in sorted(data.items()):
        ids = sorted(set(i for (c, i) in cells if c == "single") & set(i for (c, i) in cells if c == "multi"))
        if len(ids) < MIN_INSTR:
            continue
        if min(np.median([len(cells[("single", i)]) for i in ids]),
               np.median([len(cells[("multi", i)]) for i in ids])) < MIN_RUNS:
            continue
        binary = t in ("math", "actions")
        if not binary:
            continue  # "solved" is not well defined for BLEU; reported separately below

        # match analyze_repro.py, which excludes instructions with <2 runs from the mean,
        # so Table 2's Gap is exactly -dP from Table 1
        ids = [i for i in ids if len(cells[("single", i)]) >= 2 and len(cells[("multi", i)]) >= 2]
        if len(ids) < MIN_INSTR:
            continue
        s_mean = np.array([np.mean(cells[("single", i)]) for i in ids])
        m_mean = np.array([np.mean(cells[("multi", i)]) for i in ids])
        s_any = np.array([1.0 if max(cells[("single", i)]) >= THR else 0.0 for i in ids])
        m_any = np.array([1.0 if max(cells[("multi", i)]) >= THR else 0.0 for i in ids])
        det = np.array([1.0 if (min(cells[("single", i)]) >= THR and max(cells[("multi", i)]) < THR)
                        else 0.0 for i in ids])

        gap = s_mean.mean() - m_mean.mean()
        recov = (m_any.mean() - m_mean.mean()) / gap * 100 if gap > 1e-9 else float("nan")
        unrec_i = s_any - m_any
        unrec = unrec_i.mean() * 100
        idx = RNG.integers(0, len(ids), size=(2000, len(ids)))
        bs = unrec_i[idx].mean(axis=1) * 100
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append(dict(task=t, model=m, gap=gap * 100, recovered=recov, unrec=unrec,
                         unrec_lo=lo, unrec_hi=hi, det=det.mean() * 100, n=len(ids)))
        print(f"{t:10s}{m:18s}{s_mean.mean()*100:>7.1f}{m_mean.mean()*100:>7.1f}"
              f"{m_any.mean()*100:>8.1f}{recov:>8.1f}{unrec:>7.1f}"
              f"  [{lo:>5.1f},{hi:>5.1f}]{det.mean()*100:>6.1f}")

    print()
    for t in ("math", "actions"):
        r = [x for x in rows if x["task"] == t]
        if not r:
            continue
        pos = sum(1 for x in r if x["unrec_lo"] > 0)
        print(f"{t:8s} n={len(r)}  median recovered by oracle retry = "
              f"{np.median([x['recovered'] for x in r]):5.1f}%   "
              f"median UNRECOVERABLE gap = {np.median([x['unrec'] for x in r]):5.1f}pp   "
              f"cells with CI strictly above 0: {pos}/{len(r)}")
        print(f"{'':8s}  median deterministic-failure instructions = "
              f"{np.median([x['det'] for x in r]):5.1f}%")

    json.dump(rows, open(os.path.join(HERE, "recoverability_results.json"), "w",
                         encoding="utf-8"), indent=1)
    print("\nwrote recoverability_results.json")


if __name__ == "__main__":
    main()
