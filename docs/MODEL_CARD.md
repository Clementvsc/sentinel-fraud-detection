# Model card — Sentinel fraud scorer

Following the *Model Cards for Model Reporting* format (Mitchell et al., 2019).

## Overview

| | |
|---|---|
| **Task** | Binary risk score (0–1) for a card / account transaction, in real time |
| **Model** | Ensemble of three heads → single calibrated blended score:<br>• `HistGradientBoostingClassifier` (supervised) + isotonic calibration<br>• `IsolationForest` (unsupervised novelty)<br>• behavioural‑sequence divergence (smoothed unigram surprise + regime‑shift KL) |
| **Decision layer** | cost‑minimising threshold + deterministic rule engine → `ALLOW / REVIEW / CHALLENGE / BLOCK` |
| **Version** | 0.2 |
| **Owner** | hackathon project — defensive security use only |

## Intended use

- **In scope:** real‑time authorization‑time risk scoring for card‑present,
  card‑not‑present, ATM and account‑to‑account transfers; analyst triage;
  step‑up‑auth routing.
- **Out of scope:** credit decisioning, account‑opening / application fraud
  (different feature set — see the BAF dataset), AML transaction monitoring,
  any use as the *sole* basis for a legal or adverse action without human review.

## Training data

- **Demo model:** a seeded synthetic generator (`sentinel/datasets/synthetic.py`)
  — 400 customers × 90 days ≈ 140k transactions, 0.8 % fraud, 7 attack
  playbooks (4 classic + 3 adversarial), plus fraud‑*looking* legitimate noise.
  Distribution shapes are grounded in public fraud statistics; it is **not** real
  customer data.
- **Real‑data validation:** ULB / MLG Credit Card Fraud (OpenML 1597, CC‑BY),
  284,807 real transactions — see [REAL_DATA_VALIDATION.md](REAL_DATA_VALIDATION.md).
- **Feature pipeline is identical** offline and online (no train/serve skew);
  entity aggregates are built by replaying events in timestamp order.

## Evaluation

- **Protocol:** time‑ordered split — train 70 % / calibrate 15 % / **test on the
  last 15 %, out‑of‑time**. `python -m sentinel.eval` re‑runs the *full* pipeline
  on a fresh, unseen synthetic world.
- **Headline (synthetic, out‑of‑time):** detection ~80 %, false‑positive rate
  ~0.1 %, calibration ECE ~0.01, classifier ROC‑AUC ~0.99 / PR‑AUC ~0.89.
- **Headline (real ULB, out‑of‑time):** ROC‑AUC 0.92, PR‑AUC 0.65, Brier
  0.0038 → 0.00047 after calibration.
- **Adversarial:** three evasion playbooks tested explicitly — recall 68–80 %
  (rules‑only ≈ 0 %).
- **Latency:** P50 4 ms, P99 11 ms, ~210 txn/s/core (`python -m sentinel.audit`).
- **Fairness:** detection & FP rate sliced by home country, spend tier, channel
  (`python -m sentinel.audit`). No material country or spend‑tier disparity in
  the synthetic evaluation; the check flags any slice with FP rate > 1.25× the
  fairest slice for review.

## Limitations & risks

- Synthetic training data cannot capture every real fraud pattern; treat
  absolute numbers as illustrative until retrained on institutional data.
- The anomaly and sequence heads can drift as customer behaviour shifts — the
  built‑in **PSI monitor** exists to catch this; retrain on `alert`.
- Extreme class imbalance means PR‑AUC has high variance on small test slices.
- Blocking a legitimate transaction has real customer cost; the `CHALLENGE`
  tier and the analyst `REVIEW` queue exist so the model is rarely the *sole*
  actor on a decline. A blocked transaction never updates the behavioural
  profile (an attacker cannot drag "normal" toward their behaviour).
- Feedback loop can encode analyst bias; dispositions should be sampled/audited.

## Ethical considerations

- Features are behavioural aggregates, not protected attributes. Home country is
  used only for *impossible‑travel* and *foreign‑transaction* signals, never as a
  standalone risk factor; the fairness audit slices on it to check for proxy
  effects.
- Analyst‑review cases can include a grouped counterfactual reference path for the
  calibrated classifier. It is hypothetical and non‑causal; it must not be presented
  as a recommended customer action or as proof that a feature caused the score.
