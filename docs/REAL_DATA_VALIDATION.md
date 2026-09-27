# Validation on real transaction data

Sentinel's demo runs on a synthetic generator (it needs behavioural, geo, device
and entity fields that no anonymised public dataset carries). To show the **ML
core is not overfit to our own simulator**, we also train and evaluate it on a
real, published benchmark — downloaded cost-free, no account:

```bash
python -m sentinel.datasets.fetch ulb          # 284,807 real transactions from OpenML
SENTINEL_DATA=data/creditcard.csv SENTINEL_CSV_SCHEMA=ulb python -m sentinel.train
```

## Dataset

**ULB / MLG "Credit Card Fraud Detection"** — Dal Pozzolo, Caelen, Johnson &
Bontempi (2015), *Calibrating Probability with Undersampling for Unbalanced
Classification*, IEEE SSCI. OpenML dataset **1597**, licence **CC‑BY**.

- **284,807** real European card‑holder transactions, two days of September 2013
- **492** real frauds — **0.172 %** (extreme imbalance, the realistic regime)
- Features: `V1..V28` (PCA‑anonymised for privacy), `Time`, `Amount`

Because the features are PCA components, only the **calibrated GBM + isotonic +
IsolationForest** core runs here — the behavioural / entity / graph / sequence
features and the geo/velocity rules need raw fields the dataset doesn't expose.

## Results

Sentinel reports the **honest, time‑ordered** number by default: train on the
first 70 % of the two days, calibrate on the next 15 %, test on the **last 15 %,
out‑of‑time** — i.e. predict transactions that happened *after* everything the
model saw.

| Evaluation | ROC‑AUC | PR‑AUC | Brier (raw → calibrated) |
|---|---|---|---|
| **Time‑ordered, out‑of‑time** (what Sentinel prints) | **0.924** | **0.652** | 0.0038 → **0.00047** (8× better) |
| Random 5‑fold CV (what most write‑ups report) | 0.968 ± 0.013 | 0.752 ± 0.029 | — |

At the cost‑minimising threshold on the out‑of‑time slice: **precision 0.63,
recall 0.69**.

### Reading the numbers

- The **0.10 PR‑AUC gap** between random CV and time‑ordered is the point:
  shuffling lets the model peek at the future. Fraud tactics drift *within* two
  days, so the out‑of‑time score is lower — and it's the one that matters in
  production. Sentinel is built to report that one.
- **Calibration cuts Brier ~8×** on real data too — the isotonic layer isn't a
  synthetic‑data artefact.
- The OOT test slice has only ~70 frauds, so PR‑AUC has real variance; treat it
  as "mid‑0.6s", not a point estimate.

## What this does and doesn't show

✅ The model core trains, calibrates and generalises on real, imbalanced,
adversarially‑drifting data with an honest protocol.

❌ It does **not** exercise the behavioural / entity / sequence layers or the
rules — those need a dataset with raw merchant / device / geo / timestamp fields
(e.g. Kaggle **Sparkov** or **IEEE‑CIS**; adapters included, see `DATA.md`).
