# Sentinel — Model Evaluation Report

_Generated 2026-09-27T21:56:30+00:00 by `python -m sentinel.eval`. Regenerate after any model, rule or threshold change._

## Test set
- held-out synthetic world (different seed from training): seed 1041 (training seed 42), 110 customers × 35 days
- 14,151 transactions, 188 fraud (1.33% prevalence)
- Definition: positive prediction = BLOCK or CHALLENGE; positive label = fraud

## Confusion matrix (operating point)

| | Predicted fraud (stopped) | Predicted legit (allowed/review) |
|---|---:|---:|
| **Actual fraud** | 150 (TP) | 38 (FN) |
| **Actual legit** | 157 (FP) | 13,806 (TN) |

## Metrics at the operating point

| Metric | Value |
|---|---:|
| Recall / detection rate | 79.8% (95% CI 73.7%–85.6%) |
| Precision | 48.9% (95% CI 43.7%–54.3%) |
| F1 | 60.6% (95% CI 56.0%–65.6%) |
| Specificity | 98.88% (95% CI 98.69%–99.05%) |
| False-positive rate (stopped) | 1.124% (95% CI 0.953%–1.308%) |
| False-positive rate (block only) | 0.064% |
| Matthews correlation (MCC) | 0.618 |
| Balanced accuracy | 89.3% |

## Threshold-free ranking quality (calibrated probability)
- ROC-AUC: **0.8691**
- PR-AUC: **0.5982** (random baseline = prevalence = 0.01329)

## Calibration
- Expected calibration error: 0.00586 · Brier score: 0.00692

| Predicted band | n | Mean predicted | Observed fraud rate |
|---|---:|---:|---:|
| 0.0–0.1 | 13,966 | 0.001 | 0.005 |
| 0.1–0.2 | 19 | 0.100 | 0.210 |
| 0.3–0.4 | 30 | 0.336 | 0.400 |
| 0.4–0.5 | 1 | 0.403 | 0.000 |
| 0.5–0.6 | 32 | 0.500 | 0.281 |
| 0.7–0.8 | 40 | 0.741 | 0.900 |
| 0.9–1.0 | 63 | 0.969 | 0.905 |

## Recall by fraud scenario

| Scenario | Fraud | Caught | Recall |
|---|---:|---:|---:|
| account_takeover | 23 | 23 | 100.0% |
| amount_just_under (adversarial) | 27 | 22 | 81.5% |
| bust_out | 22 | 21 | 95.5% |
| card_testing | 20 | 11 | 55.0% |
| geo_consistent_ato (adversarial) | 25 | 18 | 72.0% |
| slow_drip (adversarial) | 56 | 46 | 82.1% |
| stolen_card_geo | 15 | 9 | 60.0% |

## Fairness by customer age

| Age | Transactions | Fraud | Recall | False-positive rate |
|---|---:|---:|---:|---:|
| 18-25 | 1,929 | 14 | 78.6% | 1.149% |
| 26-40 | 5,522 | 109 | 78.9% | 1.071% |
| 41-60 | 3,931 | 20 | 90.0% | 1.151% |
| 60+ | 2,769 | 45 | 77.8% | 1.175% |

- False-positive-rate gap between age groups: 0.104%
- Recall gap between age groups: 12.2%

## Business impact
- Fraud value in test set: ₹192,234; prevented: ₹164,165 (85.4%)
- Decision mix: {'ALLOW': 13673, 'REVIEW': 171, 'CHALLENGE': 204, 'BLOCK': 103}
- Score drift (PSI) vs training reference: 0.0268 once customer profiles have warmed up (second half of the run); 0.2833 over the whole run, which includes the cold-start period where every test customer is new (PSI < 0.1 stable, 0.1–0.25 watch, > 0.25 investigate)

## Limitations
- The test world is synthetic (different seed, same generator). It measures generalisation to unseen customers and fraud episodes, not to a real bank's population; validate on the bank's own labelled history before production use.
- Real-data replays (UPI, PaySim) have no customer-age column, so age-based fairness is only measured on synthetic customers.
- Confidence intervals are percentile bootstrap over transactions and do not account for correlation between transactions of the same customer or fraud episode, so they are somewhat optimistic.
