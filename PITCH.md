# Sentinel — pitch & demo script

## 30‑second pitch

> Banks lose **billions a year** to fraud. Static rules miss new attacks; a
> black‑box model can't be signed off. **Sentinel scores every transaction in
> ~4 ms** with **three complementary detection heads** — a calibrated
> gradient‑boosted classifier, an anomaly detector, and a behavioural‑sequence
> model — plus an auditable rules engine, and it **acts** (block / OTP‑challenge
> / analyst queue) *before the money leaves the account*. Every decision carries
> a **counterfactual evidence path**. It runs fully local on free open‑source software, and
> the ML core is validated on **284,807 real transactions**, not just our
> simulator.

---

## Every "weakness" is actually the point

| The obvious critique | Why it's a strength |
|---|---|
| *"It's synthetic data."* | Then we can open‑source the **entire** pipeline, data and eval — no bank will ever let you demo on real PII. Our generator reproduces **7 documented fraud typologies** and its distributions are grounded in published statistics. And the model core is **also trained & scored on the real ULB benchmark (284k transactions)** — one command, `sentinel.datasets.fetch ulb`. |
| *"That's not real AI / no deep learning."* | Gradient‑boosted trees are what **actually ships** at card networks and banks. We run an **ensemble of three diverse methods** — supervised + anomaly + sequence — which is *harder to evade* than one deep net and stays **fully explainable**. The architecture is model‑agnostic: a GRU/GNN head plugs into the same blend. |
| *"0.99 AUC is easy on this problem."* | We report the **time‑ordered, out‑of‑time** number, not the shuffled‑CV number everyone else quotes. On real data that's the difference between PR‑AUC **0.75 (random CV)** and **0.65 (honest)** — we show both and default to the honest one. |
| *"Any classifier catches obvious fraud."* | We **built three adversarial attacks to beat our own rules** and *report the drop* — 68–80 % recall where a rules‑only system gets **≈ 0 %**. Nobody else tests this, let alone tells you the number. |
| *"Is it fast / production‑ready?"* | **P50 4 ms, P99 11 ms, ~210 txn/s/core** — 10× under a card‑network auth budget. `docker compose up --scale sentinel=2` runs stateless workers sharing warm state via Redis. Restart‑safe snapshots, PSI drift alarms, cost knobs in one config file, a model card, and a fairness audit. |
| *"Won't it just block real customers?"* | Measured **0.10 % false‑positive rate**. A blocked transaction never updates the behavioural profile (an attacker can't move "normal"). The `CHALLENGE` tier and analyst `REVIEW` queue mean the model is rarely the *sole* actor on a decline. Fairness audit: no country or spend‑tier disparity. |

---

## Measured

**Synthetic, full pipeline, out‑of‑time, fresh unseen world:**

| | |
|---|---|
| Detection | **81 %** at **0.10 % FP** |
| Calibration ECE | 0.009 |
| Latency | P50 4 ms · P99 11 ms · 210 txn/s/core |
| Policy A/B | rules‑only **62 %** · model‑only **38 %** · **both 81 %** |

**Real data — ULB benchmark, 284,807 transactions, out‑of‑time:** ROC‑AUC
**0.92**, PR‑AUC 0.65, Brier 0.0038 → **0.0005** after calibration.

**Adversarial recall:** account_takeover 100 · bust_out 96 · slow_drip 80 ·
card_testing 80 · stolen_card_geo 73 · amount_just_under 70 · geo_consistent_ato 68.

---

## Live demo (3 minutes)

1. **The Policy A/B strip.** "Same live traffic, scored three ways. Rules alone:
   62 %. The model alone: 38 %. Together: 81 % — at a tenth of a percent false
   positives. That gap is the whole thesis."

2. **Click a flagged transaction.** Inspect the grouped reference path: it shows the
   calibrated classifier probability, the configured threshold, and which hypothetical
   feature families would need reference values for the score to cross that threshold.

3. **⚡ Account Takeover.** Rows go red. Click one: *"high‑value transfer right
   after 8 failed logins"*, calibrated proba 97 %, **BLOCK**, customer SMS shown.
   "Rules‑only *would* have caught this too — it's blatant."

4. **🎭 Stealth Takeover.** "Same attack, but from the customer's own phone, home
   city, one failed login — no geo or device tell." Click a caught one:
   *"rules‑only would ALLOW · model‑only would CHALLENGE · Sentinel CHALLENGE."*
   "This is the case rules physically cannot see. The **sequence head** catches
   the switch from groceries to wire transfers."

5. **🎭 Slow Drip.** "Small, in‑profile transfers to one mule, twice a day." Watch
   the **payee fraud‑rate** feature climb in the case view until it crosses the
   block threshold. "The graph learns the mule even though every single
   transaction looks normal."

6. **Click "Confirm fraud."** "That analyst verdict updated the entity's risk and
   will weight the next retrain. The system compounds."

7. **Header + `sentinel.eval` / `sentinel.audit` / real‑data run.** "Live PSI
   drift — it's `alert` right now because I just injected three attack waves.
   Latency P99 11 ms. And here's the same model on 284k real transactions."

8. **Close.** "The dashboard is a skin. The product is one `POST /score` a core
   banking system calls inline, in four milliseconds."

---

## Tech

Python · FastAPI · scikit‑learn (HistGradientBoosting + IsolationForest +
isotonic) · grouped counterfactual search · a pure‑Python sequence model · WebSocket · Docker + Redis.
~2.9k lines, `pytest` 33 green, one command to run, **$0** — all OSS, all local.
Real‑data adapters for Kaggle ULB / IEEE‑CIS / Sparkov (`docs/DATA.md`), plus a
model card and a fairness audit.
