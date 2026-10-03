# -*- coding: utf-8 -*-
"""Reproduce + EXTEND "LLMs Get Lost In Multi-Turn Conversation" (ICLR 2026 Outstanding).

REPRODUCTION: compute their three metrics on our own Bedrock-generated conversations.
  P     = mean(S)                                    average performance
  A90   = percentile90(S)                            aptitude ("best case")
  U90_10= percentile90(S) - percentile10(S)          unreliability (interpercentile range)
(definitions quoted from arXiv 2505.06120)

EXTENSION (the contribution): their headline decomposition - "the multi-turn drop is a MINOR loss
in aptitude and a LARGE increase in unreliability" - depends on an UNDECLARED choice of the 90th
and 10th percentiles. We sweep that convention (95/5, 90/10, 80/20, 75/25) and report whether the
qualitative conclusion survives. Bootstrap CIs over instructions.

Run: python analyze_repro.py
"""
import glob, json, os, statistics, sys
import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
OUT = os.path.dirname(os.path.abspath(__file__))
CONVENTIONS = [(95, 5), (90, 10), (80, 20), (75, 25)]
RNG = np.random.default_rng(0)


def load(logdir):
    """-> {conv_type: {task_id: [scores]}}"""
    out = {}
    for p in glob.glob(os.path.join(REPO, logdir, "**", "*.jsonl"), recursive=True):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            s = d.get("score")
            if s is None:
                continue
            ct = "multi-turn" if str(d.get("conv_type", "")).startswith("sharded") else "single-turn"
            m = d.get("assistant_model", "unknown")
            tk = d.get("task", "unknown")
            key = f"{m} | {tk}"
            out.setdefault(key, {}).setdefault(ct, {}).setdefault(d.get("task_id"), []).append(float(s) * 100.0)
    return out


def usable_ids(single, multi, min_runs=2):
    """Instructions with enough runs in BOTH conditions.

    The two conditions must be evaluated on the SAME instruction set, otherwise dA and dU are
    differences of means over different populations. (Panel review caught this: one cell had
    19 single-turn vs 20 multi-turn instructions feeding a single reported delta.)
    """
    return {t for t in set(single) & set(multi)
            if len(single[t]) >= min_runs and len(multi[t]) >= min_runs}


def metrics(per_instr, hi, lo, ids=None):
    """Their per-instruction metrics, averaged over instructions (corpus level).

    `ids` pins the instruction set so both conditions are scored on identical instructions.
    """
    Ps, As, Us = [], [], []
    for tid, S in per_instr.items():
        if ids is not None and tid not in ids:
            continue
        if len(S) < 2:
            continue
        Ps.append(np.mean(S))
        As.append(np.percentile(S, hi))
        Us.append(np.percentile(S, hi) - np.percentile(S, lo))
    return (float(np.mean(Ps)) if Ps else 0.0,
            float(np.mean(As)) if As else 0.0,
            float(np.mean(Us)) if Us else 0.0,
            len(Ps))


def per_instruction(per_instr, hi, lo, ids):
    """Per-instruction (P, A, U) arrays aligned to ids."""
    P, A, U = [], [], []
    for t in ids:
        S = per_instr[t]
        P.append(np.mean(S)); A.append(np.percentile(S, hi))
        U.append(np.percentile(S, hi) - np.percentile(S, lo))
    return np.array(P), np.array(A), np.array(U)


def boot_delta(single, multi, hi, lo, B=2000):
    """Bootstrap over instructions. Per-instruction stats are computed ONCE, then resampled -
    this is the correct instruction-level bootstrap and is orders of magnitude faster."""
    ids = sorted(set(single) & set(multi))
    Ps, As, Us = per_instruction(single, hi, lo, ids)
    Pm, Am, Um = per_instruction(multi, hi, lo, ids)
    dP_i, dA_i, dU_i = Pm - Ps, Am - As, Um - Us
    n = len(ids)
    idx = RNG.integers(0, n, size=(B, n))
    dP = dP_i[idx].mean(axis=1); dA = dA_i[idx].mean(axis=1); dU = dU_i[idx].mean(axis=1)
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return q(dP), q(dA), q(dU)


def main(logdir="logs_run1", model=None):
    bymodel = load(logdir)
    models = [model] if model else sorted(bymodel)
    all_out = {}
    for mdl in models:
        data = bymodel.get(mdl, {})
        if "single-turn" not in data or "multi-turn" not in data:
            print(f"[skip {mdl}] need both conversation types; found {list(data)}")
            continue
        shared_ids = set(data["single-turn"]) & set(data["multi-turn"])
        if len(shared_ids) < 10:
            print(f"[skip {mdl}] only {len(shared_ids)} paired instructions - under-powered")
            continue
        # percentiles are taken WITHIN an instruction, so runs-per-instruction is the binding
        # constraint. Require a median of >= MIN_RUNS in both conditions.
        MIN_RUNS = 4
        import statistics as _st
        med_s = _st.median(len(data["single-turn"][i]) for i in shared_ids)
        med_m = _st.median(len(data["multi-turn"][i]) for i in shared_ids)
        if min(med_s, med_m) < MIN_RUNS:
            print(f"[skip {mdl}] median runs/instruction = {min(med_s, med_m):.0f} "
                  f"(< {MIN_RUNS}) - percentiles within an instruction are not estimable")
            continue
        print(chr(10)+"########## MODEL: "+str(mdl))
        all_out[mdl] = _analyze(data, logdir, mdl)
    json.dump(all_out, open(os.path.join(OUT, "repro_results_bymodel.json"), "w"), indent=1)
    print(chr(10)+"wrote repro_results_bymodel.json ("+str(len(all_out))+" models)")
    return 0


def _safe(name):
    """Windows-safe filename fragment (the cell key contains ' | ')."""
    import re as _re
    return _re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")


def _analyze(data, logdir, mdl):
    single, multi = data["single-turn"], data["multi-turn"]
    shared = sorted(set(single) & set(multi))
    single = {k: single[k] for k in shared}
    multi = {k: multi[k] for k in shared}
    nconv = sum(len(v) for v in single.values()) + sum(len(v) for v in multi.values())
    print(f"REPRODUCTION on our own Bedrock runs | {len(shared)} instructions, {nconv} conversations\n")

    rows = []
    for hi, lo in CONVENTIONS:
        _ids = usable_ids(single, multi)
        Ps, As, Us, n = metrics(single, hi, lo, ids=_ids)
        Pm, Am, Um, n_m = metrics(multi, hi, lo, ids=_ids)
        assert n == n_m, f"unpaired instruction sets: {n} vs {n_m}"
        dP_ci, dA_ci, dU_ci = boot_delta(single, multi, hi, lo)
        rows.append(dict(convention=f"A{hi}/U{hi}_{lo}", hi=hi, lo=lo,
                         P_single=Ps, P_multi=Pm, dP=Pm - Ps, dP_ci=dP_ci,
                         A_single=As, A_multi=Am, dA=Am - As, dA_ci=dA_ci,
                         U_single=Us, U_multi=Um, dU=Um - Us, dU_ci=dU_ci,
                         n_instructions=n))

    print(f"{'convention':14s} {'dPerf':>8s} {'dAptitude':>22s} {'dUnreliability':>24s}")
    for r in rows:
        a = f"{r['dA']:+6.1f} [{r['dA_ci'][0]:+.1f},{r['dA_ci'][1]:+.1f}]"
        u = f"{r['dU']:+6.1f} [{r['dU_ci'][0]:+.1f},{r['dU_ci'][1]:+.1f}]"
        print(f"{r['convention']:14s} {r['dP']:+8.1f} {a:>22s} {u:>24s}")

    base = next(r for r in rows if r["hi"] == 90)
    print(f"\nPAPER'S CONVENTION (A90, U90_10): aptitude {base['dA']:+.1f}pp, "
          f"unreliability {base['dU']:+.1f}pp, performance {base['dP']:+.1f}pp")
    dA_range = (min(r["dA"] for r in rows), max(r["dA"] for r in rows))
    dU_range = (min(r["dU"] for r in rows), max(r["dU"] for r in rows))
    print(f"ACROSS CONVENTIONS: aptitude delta spans {dA_range[0]:+.1f} to {dA_range[1]:+.1f}pp; "
          f"unreliability delta spans {dU_range[0]:+.1f} to {dU_range[1]:+.1f}pp")
    # does the qualitative claim (|dU| >> |dA|) hold under every convention?
    holds = [abs(r["dU"]) > abs(r["dA"]) for r in rows]
    print(f"Qualitative claim 'unreliability dominates aptitude' holds under "
          f"{sum(holds)}/{len(holds)} conventions.")

    result = dict(logdir=logdir, model=mdl, n_instructions=len(shared), n_conversations=nconv,
                  rows=rows, claim_holds=sum(holds), n_conventions=len(holds))
    json.dump(result, open(os.path.join(OUT, f"repro_results_{_safe(mdl)}.json"), "w"), indent=1)

    # ---- figure ----
    # viz_style.py ships alongside this file
    import viz_style as vs
    vs.apply()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(vs.WIDE, 2.8))
    x = np.arange(len(rows)); w = 0.38
    dA = [r["dA"] for r in rows]; dU = [r["dU"] for r in rows]
    aerr = [[max(0.0, r["dA"] - r["dA_ci"][0]) for r in rows], [max(0.0, r["dA_ci"][1] - r["dA"]) for r in rows]]
    uerr = [[max(0.0, r["dU"] - r["dU_ci"][0]) for r in rows], [max(0.0, r["dU_ci"][1] - r["dU"]) for r in rows]]
    ax.bar(x - w/2, dA, w, yerr=aerr, capsize=2, error_kw={"lw": 0.7},
           color=vs.OKABE_ITO["blue"], label="change in aptitude")
    ax.bar(x + w/2, dU, w, yerr=uerr, capsize=2, error_kw={"lw": 0.7},
           color=vs.OKABE_ITO["orange"], label="change in unreliability")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([r["convention"] for r in rows])
    ax.set_ylabel("multi-turn minus single-turn (pp)")
    vs.legend_above(ax, 2)
    vs.finish(ax)
    cap = ("Figure R1: Reproduction and convention sweep of the aptitude/unreliability decomposition "
           "from Laban et al. (ICLR 2026 Outstanding Paper). Single- vs multi-turn conversations "
           f"simulated on {mdl} via AWS Bedrock ({len(shared)} GSM8K-derived instructions, "
           f"{nconv} conversations, 6 runs per instruction per condition; bootstrap 95\\% CIs over "
           "instructions). The paper's metrics use the 90th/10th percentiles; we sweep that undeclared "
           "choice. Caveat: one open-weight model and one task, so absolute levels are not comparable "
           "to the paper's closed-model results.")
    vs.save(fig, os.path.join(OUT, "figR1_convention_sweep_" + _safe(mdl)), cap)
    print(f"\nwrote repro_results.json + figR1 to {OUT}")
    return result


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "logs_run1", sys.argv[2] if len(sys.argv) > 2 else None))
