"""Print Table 1 and Table 2 of the paper from the JSON results.

Run analyze_repro.py and analyze_recoverability.py first (or use the shipped JSON).
Offline: no model or API call is made.

Usage:
    python paper_tables.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = {"math": "Mathematics", "actions": "Function calling", "data2text": "Data-to-text"}

print("Table 1: decomposition at the 90/10 convention (repro_results_bymodel.json)")
print("%-17s %-18s %6s %7s %7s %7s %7s" % ("task", "model", "U_s", "dP", "dA", "dU", "m"))
by = json.load(open(os.path.join(HERE, "repro_results_bymodel.json"), encoding="utf-8"))
rows = []
for key, v in by.items():
    model, task = [s.strip() for s in key.split("|")]
    r = [x for x in v["rows"] if x["convention"] == "A90/U90_10"][0]
    rows.append((list(TASK).index(task), -(r["dU"] - abs(r["dA"])), task, model, r))
for _, _, task, model, r in sorted(rows):
    print("%-17s %-18s %6.1f %+7.1f %+7.1f %+7.1f %+7.1f"
          % (TASK[task], model, r["U_single"], r["dP"], r["dA"], r["dU"], r["dU"] - abs(r["dA"])))

print("\nTable 2: operational retry test (recoverability_results.json)")
print("%-17s %-18s %6s %8s %7s %16s %7s" % ("task", "model", "gap", "recov%", "unrec", "95% CI", "determ"))
rec = json.load(open(os.path.join(HERE, "recoverability_results.json"), encoding="utf-8"))
for r in sorted(rec, key=lambda x: (list(TASK).index(x["task"]), -x["recovered"])):
    print("%-17s %-18s %6.1f %8.1f %7.1f   [%5.1f, %5.1f] %7.1f"
          % (TASK[r["task"]], r["model"], r["gap"], r["recovered"], r["unrec"],
             r["unrec_lo"], r["unrec_hi"], r["det"]))
