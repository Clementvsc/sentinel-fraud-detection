"""Honest evaluation on a fresh, unseen synthetic world.

    python -m sentinel.eval            # default held-out world
    python -m sentinel.eval --quick    # smaller world, for a fast sanity check

Runs every transaction of a held-out world (different seed from training)
through the *full* production pipeline and writes:

    sentinel/reports/eval_report.json   machine-readable (served at /evaluation)
    docs/EVALUATION.md                  human-readable model-risk summary

See sentinel/evaluation.py for the exact metric definitions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import config
from .evaluation import evaluate, save_report

DOC_PATH = config.ROOT.parent / "docs" / "EVALUATION.md"


def _p(x, d=1):
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


def _num(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def _ci(r, k, d=1):
    c = r["confidence_intervals_95"].get(k)
    return "" if not c else f" (95% CI {_p(c[0], d)}–{_p(c[1], d)})"


def to_markdown(r: dict) -> str:
    m, cm, ds = r["metrics"], r["confusion_matrix"], r["dataset"]
    L = [
        "# Sentinel — Model Evaluation Report",
        "",
        f"_Generated {r['generated_at']} by `python -m sentinel.eval`. "
        "Regenerate after any model, rule or threshold change._",
        "",
        "## Test set",
        f"- {ds['kind']}: seed {ds['seed']} (training seed {ds['train_seed']}), "
        f"{ds['customers']} customers × {ds['days']} days",
        f"- {ds['transactions']:,} transactions, {ds['fraud']} fraud "
        f"({_p(ds['fraud_prevalence'], 2)} prevalence)",
        f"- Definition: {r['definition']}",
        "",
        "## Confusion matrix (operating point)",
        "",
        "| | Predicted fraud (stopped) | Predicted legit (allowed/review) |",
        "|---|---:|---:|",
        f"| **Actual fraud** | {cm['tp']:,} (TP) | {cm['fn']:,} (FN) |",
        f"| **Actual legit** | {cm['fp']:,} (FP) | {cm['tn']:,} (TN) |",
        "",
        "## Metrics at the operating point",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Recall / detection rate | {_p(m['recall'])}{_ci(r, 'recall')} |",
        f"| Precision | {_p(m['precision'])}{_ci(r, 'precision')} |",
        f"| F1 | {_p(m['f1'])}{_ci(r, 'f1')} |",
        f"| Specificity | {_p(m['specificity'], 2)}{_ci(r, 'specificity', 2)} |",
        f"| False-positive rate (stopped) | {_p(m['false_positive_rate'], 3)}{_ci(r, 'false_positive_rate', 3)} |",
        f"| False-positive rate (block only) | {_p(r['block_only']['false_positive_rate'], 3)} |",
        f"| Matthews correlation (MCC) | {_num(m['mcc'])} |",
        f"| Balanced accuracy | {_p(m['balanced_accuracy'])} |",
        "",
        "## Threshold-free ranking quality (calibrated probability)",
        f"- ROC-AUC: **{r['threshold_free']['roc_auc']}**",
        f"- PR-AUC: **{r['threshold_free']['pr_auc']}** "
        f"(random baseline = prevalence = {r['threshold_free']['pr_auc_baseline']})",
        "",
        "## Calibration",
        f"- Expected calibration error: {r['calibration']['ece']} · Brier score: {r['calibration']['brier']}",
        "",
        "| Predicted band | n | Mean predicted | Observed fraud rate |",
        "|---|---:|---:|---:|",
    ]
    for row in r["calibration"]["table"]:
        L.append(f"| {row['lo']:.1f}–{row['hi']:.1f} | {row['n']:,} | {row['predicted']:.3f} | {row['observed']:.3f} |")
    L += ["", "## Recall by fraud scenario", "", "| Scenario | Fraud | Caught | Recall |", "|---|---:|---:|---:|"]
    for s, v in r["per_scenario"].items():
        tag = " (adversarial)" if v["adversarial"] else ""
        L.append(f"| {s}{tag} | {v['fraud']} | {v['caught']} | {_p(v['recall'])} |")
    L += ["", "## Fairness by customer age", "",
          "| Age | Transactions | Fraud | Recall | False-positive rate |", "|---|---:|---:|---:|---:|"]
    for a in r["per_age_bracket"]:
        if not a.get("transactions"):
            L.append(f"| {a['bracket']} | 0 | – | – | – |")
            continue
        L.append(f"| {a['bracket']} | {a['transactions']:,} | {a['fraud']} | "
                 f"{_p(a['recall'])} | {_p(a['false_positive_rate'], 3)} |")
    f = r["fairness"]
    L += ["",
          f"- False-positive-rate gap between age groups: {_p(f['false_positive_rate_gap'], 3)}",
          f"- Recall gap between age groups: {_p(f['recall_gap'])}",
          "",
          "## Business impact",
          f"- Fraud value in test set: ₹{r['money']['fraud_amount_inr']:,.0f}; "
          f"prevented: ₹{r['money']['prevented_inr']:,.0f} ({_p(r['money']['prevented_share'])})",
          f"- Decision mix: {r['decision_mix']}",
          f"- Score drift (PSI) vs training reference: {r['drift_psi_after_warmup']} once customer "
          f"profiles have warmed up (second half of the run); {r['drift_psi_vs_training']} over the "
          "whole run, which includes the cold-start period where every test customer is new "
          "(PSI < 0.1 stable, 0.1–0.25 watch, > 0.25 investigate)",
          "",
          "## Limitations",
          "- The test world is synthetic (different seed, same generator). It measures generalisation "
          "to unseen customers and fraud episodes, not to a real bank's population; validate on the "
          "bank's own labelled history before production use.",
          "- Real-data replays (UPI, PaySim) have no customer-age column, so age-based fairness is only "
          "measured on synthetic customers.",
          "- Confidence intervals are percentile bootstrap over transactions and do not account for "
          "correlation between transactions of the same customer or fraud episode, so they are "
          "somewhat optimistic.",
          ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small world, fewer bootstrap samples")
    a = ap.parse_args()
    kw = dict(n_customers=40, days=20, n_boot=100) if a.quick else {}
    print("  scoring the held-out world through the full pipeline ...")
    r = evaluate(progress=lambda d, n: print(f"    {d:,}/{n:,}"), **kw)
    p = save_report(r)
    md = to_markdown(r)
    if not a.quick:
        DOC_PATH.write_text(md)
    print(md)
    print(f"\n  wrote {p}" + ("" if a.quick else f" and {DOC_PATH}"))


if __name__ == "__main__":
    main()
