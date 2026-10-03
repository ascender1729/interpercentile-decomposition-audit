# -*- coding: utf-8 -*-
"""Combined cross-family figure: does the aptitude/unreliability decomposition depend on the
percentile convention in the SAME way across three model families?
Reads repro_results_bymodel.json produced by analyze_repro.py."""
import json, os, sys
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
# viz_style.py ships alongside this file
import viz_style as vs
vs.apply()
import matplotlib.pyplot as plt

data = json.load(open(os.path.join(OUT, "repro_results_bymodel.json"), encoding="utf-8"))
models = [m for m, v in data.items() if v and v.get("n_instructions", 0) >= 5]
if not models:
    raise SystemExit("no models with enough data yet")

convs = [r["convention"] for r in data[models[0]]["rows"]]
fig, axes = plt.subplots(1, len(models), figsize=(vs.WIDE, 2.6), sharey=True)
if len(models) == 1:
    axes = [axes]

for ax, m in zip(axes, models):
    rows = data[m]["rows"]
    x = np.arange(len(rows)); w = 0.38
    dA = [r["dA"] for r in rows]; dU = [r["dU"] for r in rows]
    aerr = [[r["dA"] - r["dA_ci"][0] for r in rows], [r["dA_ci"][1] - r["dA"] for r in rows]]
    uerr = [[r["dU"] - r["dU_ci"][0] for r in rows], [r["dU_ci"][1] - r["dU"] for r in rows]]
    ax.bar(x - w/2, dA, w, yerr=aerr, capsize=1.5, error_kw={"lw": 0.6},
           color=vs.OKABE_ITO["blue"], label="aptitude")
    ax.bar(x + w/2, dU, w, yerr=uerr, capsize=1.5, error_kw={"lw": 0.6},
           color=vs.OKABE_ITO["orange"], label="unreliability")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([c.split("/")[0] for c in convs], rotation=45, ha="right", fontsize=6.5)
    holds = data[m]["claim_holds"]; nc = data[m]["n_conventions"]
    ax.set_title(f"{m}\n(claim holds {holds}/{nc})", fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)

axes[0].set_ylabel("multi-turn minus single-turn (pp)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower left", bbox_to_anchor=(0.08, 0.96, 0.86, 0.1), mode="expand",
           ncol=2, frameon=False, fontsize=7)
fig.tight_layout(rect=[0, 0, 1, 0.93])

ns = {m: (data[m]["n_instructions"], data[m]["n_conversations"]) for m in models}
cap = ("Figure R2: The aptitude/unreliability decomposition of Laban et al. (ICLR 2026 Outstanding "
       "Paper) is sensitive to their undeclared percentile convention, across three model families. "
       "Each panel sweeps the convention from A95/U95-5 to A75/U75-25 on identical GSM8K-derived "
       "sharded instructions simulated via AWS Bedrock "
       + "; ".join(f"{m}: {v[0]} instructions, {v[1]} conversations" for m, v in ns.items())
       + ". Bars are multi-turn minus single-turn with bootstrap 95\\% CIs over instructions. "
       "Average performance is invariant to the convention; the split between aptitude loss and "
       "unreliability gain is not. Caveat: one task (math, deterministic scorer), open-weight models "
       "only, so absolute levels are not comparable to the paper's closed-model results.")
vs.save(fig, os.path.join(OUT, "figR2_crossfamily_convention"), cap)
print("models:", models)
for m in models:
    print(f"  {m}: claim holds {data[m]['claim_holds']}/{data[m]['n_conventions']}, "
          f"n_instr={data[m]['n_instructions']}, n_conv={data[m]['n_conversations']}")
print("wrote figR2_crossfamily_convention.pdf/.png")
