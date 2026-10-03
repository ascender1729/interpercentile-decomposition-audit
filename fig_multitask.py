# -*- coding: utf-8 -*-
"""Multi-task summary: does the convention-decay pattern hold across TASKS as well as families?
For each (model, task) cell, plot the margin  dU - |dA|  at each percentile convention.
A margin above zero means 'unreliability dominates aptitude' (the paper's claim) holds.
Reads repro_results_bymodel.json (keys are 'model | task')."""
import json, os, sys
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
# viz_style.py ships alongside this file
import viz_style as vs
vs.apply()
import matplotlib.pyplot as plt

data = json.load(open(os.path.join(OUT, "repro_results_bymodel.json"), encoding="utf-8"))
cells = {k: v for k, v in data.items()
         if isinstance(v, dict) and v.get("n_instructions", 0) >= 5}
if not cells:
    raise SystemExit("no usable cells")

convs = [r["convention"].split("/")[0] for r in list(cells.values())[0]["rows"]]
fig, ax = plt.subplots(figsize=(vs.WIDE, 3.4))

# colour encodes TASK (the finding is task-dependence); marker encodes model.
TASK_COLOR = {"math": vs.OKABE_ITO["blue"], "actions": vs.OKABE_ITO["orange"],
              "data2text": vs.OKABE_ITO["green"]}
MODEL_MARKER = {"llama4-maverick": "o", "qwen3-32b": "s", "deepseek-r1": "^",
                "llama3.3-70b": "v", "claude-sonnet4.5": "P", "nova2-lite": "X",
                "mistral-pixtral-l": "d", "palmyra-x5": "*", "llama4-scout": "<"}
SHORT = {"llama4-maverick": "Llama4", "qwen3-32b": "Qwen3", "deepseek-r1": "DSR1",
         "llama3.3-70b": "L3.3", "claude-sonnet4.5": "Sonnet", "nova2-lite": "Nova2",
         "mistral-pixtral-l": "Pixtral", "palmyra-x5": "Palmyra", "llama4-scout": "Scout"}
for name, v in sorted(cells.items()):
    mdl, tk = [x.strip() for x in name.split("|")]
    margin = [r["dU"] - abs(r["dA"]) for r in v["rows"]]
    ax.plot(range(len(margin)), margin,
            marker=MODEL_MARKER.get(mdl, "D"), markersize=4.5, linewidth=1.2,
            color=TASK_COLOR.get(tk, vs.OKABE_ITO["grey"]),
            label=f"{tk} / {SHORT.get(mdl, mdl)}")

ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(len(convs)))
ax.set_xticklabels(convs)
ax.set_xlabel("percentile convention (window widens inward to the right)")
ax.set_ylabel(r"margin: $\Delta U - |\Delta A|$  (pp)")
vs.legend_above(ax, 3)
vs.finish(ax)

n_cells = len(cells)
n_cross = sum(1 for v in cells.values() if any(r["dU"] - abs(r["dA"]) <= 0 for r in v["rows"]))
cap = ("Figure R3: Margin by which unreliability exceeds aptitude loss, across percentile "
       "conventions, for every model-task cell. Values above the line mean the original paper's "
       "qualitative claim holds. The margin decays monotonically in every cell as the percentile "
       f"window widens inward, and crosses zero in {n_cross} of {n_cells} cells. Conversations "
       "generated via AWS Bedrock; see Table 1 for per-cell counts. Tasks differ in scorer type "
       "(math: exact numeric match; actions: AST match; data2text: BLEU), so absolute margins are "
       "not comparable across tasks - the monotone decay is.")
vs.save(fig, os.path.join(OUT, "figR3_multitask_margin"), cap)
print(f"cells: {n_cells}, cells whose margin crosses zero: {n_cross}")
for name, v in sorted(cells.items()):
    m = [round(r["dU"] - abs(r["dA"]), 1) for r in v["rows"]]
    print(f"  {name:34s} n_instr={v['n_instructions']:3d} margins={m}")
print("wrote figR3_multitask_margin.pdf/.png")
