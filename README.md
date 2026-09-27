# 🛡️ Sentinel — AI‑Powered Banking Fraud Detection

**CDT‑04 · FinTech · Cybersecurity & Digital Trust**

Real‑time detection **and response** for banking fraud. Sentinel scores every
transaction in a few milliseconds against the customer's own behaviour *and* the
risk history of the merchant / device / payee it touches, then **blocks or
step‑up‑challenges** suspicious activity before the money moves — with
plain‑English evidence and a hypothetical counterfactual path for analyst review.

Everything here is **free and fully local** — open‑source libraries only, no paid
APIs, no accounts, one command to run.

---

## What's in the box

**Three complementary detection heads**, blended into one calibrated score — an
ensemble of *diverse methods* is harder to evade than any single deep net, and
every head is independently explainable:

| Head | Implementation | Catches |
|---|---|---|
| **Supervised** | `HistGradientBoostingClassifier` + **isotonic calibration** (Brier ↓ ~5–8×) | known fraud shapes |
| **Anomaly** | `IsolationForest` | novel patterns with no training examples |
| **Behavioural sequence** | smoothed per‑customer unigram surprise + regime‑shift KL divergence over discrete behaviour tokens | a grocery/POS customer suddenly wiring money; drum‑beat repetition |

fed by:

| Feature family | Detail |
|---|---|
| **Behavioural** (31) | amount vs personal norm, tempo, geo, device, login history, impossible‑travel speed |
| **Entity + graph** (13, [`entities.py`](sentinel/entities.py)) | Bayesian‑smoothed, time‑decayed **historical fraud rate** per merchant / card BIN / device / payee, plus **graph fan‑in/out** (a mule collecting from 6 victims, one device on 6 logins → ring) |
| **Sequence** (4) | see above |

wrapped in:

| Layer | Implementation |
|---|---|
| **Rules** | deterministic tripwires — `known_bad_entity`, `fraud_ring`, impossible travel, velocity, login‑then‑drain |
| **Decision engine** | blended risk + **cost‑minimising threshold** (learned FN vs FP cost) → `ALLOW / REVIEW / CHALLENGE / BLOCK` + customer alert |
| **Explainability** | model-agnostic grouped counterfactual search plus clearly labeled, non-additive feature probes |
| **Drift monitoring** | live **PSI** of the risk‑score distribution vs a training reference → `warming / stable / watch / alert` |
| **Feedback loop** | `POST /feedback` records analyst dispositions + delayed chargebacks; retrain folds them in with **label‑maturity sample weighting** |
| **Policy A/B** | every transaction is *also* scored rules‑only and model‑only, live, so you can see the ensemble beat either alone |
| **State** | `StateStore` abstraction — in‑memory + atomic disk snapshot, restart‑safe; optional shared **Redis** via `SENTINEL_REDIS_URL` |
| **Assurance** | `python -m sentinel.audit` — latency percentiles + throughput, and fairness / disparate‑impact by country / spend tier / channel |
| **Data** | seeded synthetic generator (7 playbooks, 3 adversarial) **or** real data — `python -m sentinel.datasets.fetch ulb` pulls **284,807 real transactions** cost‑free, or point `SENTINEL_DATA=` at any Kaggle ULB / IEEE‑CIS / Sparkov CSV |

---

## Pipeline

```
                 ┌──────────────────────── Sentinel engine ────────────────────────┐
 transaction ───▶│ behavioural features ─┐                                          │
 + login events  │ entity / graph feats ─┼─▶ calibrated GBM ─┐                       │
                 │                        │   IsolationForest ─┼─▶ risk ─▶ decision ──┼─▶ ALLOW
                 │ deterministic rules ───┴───────────────────┘   (cost-based)       │─▶ REVIEW   → analyst queue
                 │                                                                   │─▶ CHALLENGE → OTP + alert
                 │ update profile + entity registry (if not blocked)                 │─▶ BLOCK     → stop + alert
                 │ feed drift/PSI monitor · snapshot state every N txns               │
                 └───────────────────────────────────────────────────────────────────┘
        feedback:  chargeback / analyst verdict ─▶ entity fraud counters + next retrain
```

No train/serve skew: the identical `compute_features()` runs offline and online;
entity aggregates are built by replaying events in time order.

---

## Model performance

Trained on a seeded synthetic world (**400 customers × 90 days ≈ 140k
transactions**, 0.75 % fraud) with 7 attack playbooks — including three
**adversarial** ones built to dodge naive rules (`amount_just_under`,
`slow_drip`, `geo_consistent_ato`) — plus fraud‑*looking* legitimate noise.

Validation is **time‑ordered**: train = first 70 % by timestamp, validation =
next 15 % (calibration + threshold), **test = the last 15 %, out‑of‑time**.

**Classifier, out‑of‑time test slice:**

| Metric | Value |
|---|---|
| ROC‑AUC | **0.996** |
| PR‑AUC | **0.89** |
| Precision / recall @ cost‑optimal threshold | 0.86 / 0.92 |
| Brier — raw → **calibrated** | 0.0020 → **0.0013** |

**Full pipeline** (3 heads + rules + decision) on a *fresh, unseen world*
(`python -m sentinel.eval`):

| | Value |
|---|---|
| Detection rate (BLOCK/CHALLENGE on fraud) | **81 %** |
| False‑positive rate (legit blocked) | **0.10 %** |
| Calibration ECE | **0.009** |
| PSI (risk blend) vs training reference | 0.013 (stable) |
| Latency P50 / P99 (`sentinel.audit`) | **4 ms / 11 ms**, ~210 txn/s/core |

**Policy A/B on the same traffic** — neither layer alone gets you there:

| policy | detection | FP rate |
|---|---|---|
| rules only | 62 % | 0.10 % |
| model only | 38 % | 0.00 % |
| **Sentinel (both)** | **81 %** | 0.10 % |

Per‑scenario recall — classic attacks caught cleanly; the three **adversarial**
playbooks, built specifically to evade the rules, degrade *honestly* (a
rules‑only system catches ≈ 0 % of these):

| scenario | recall |
|---|---|
| account_takeover | 100 % |
| bust_out | 96 % |
| slow_drip *(adv.)* | 80 % |
| card_testing | 80 % |
| stolen_card_geo | 73 % |
| amount_just_under *(adv.)* | 70 % |
| geo_consistent_ato *(adv.)* | 68 % |

**Fairness** (`sentinel.audit`): no material disparity in detection or
false‑positive rate across home country or customer spend tier; the check flags
any slice with FP rate > 1.25× the fairest slice for review.

> Synthetic training data — the numbers show the pipeline works and is honestly
> evaluated, not a production claim. The ML core is **also validated on real
> data**: `python -m sentinel.datasets.fetch ulb` → 284,807 real transactions →
> out‑of‑time ROC‑AUC **0.92**, calibrated Brier **0.0005**. See
> [docs/REAL_DATA_VALIDATION.md](docs/REAL_DATA_VALIDATION.md) and
> [docs/DATA.md](docs/DATA.md) · [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

---

## Quick start on Windows

1. Install **Python 3.11 or newer** (from python.org) if it is not already installed. Keep an internet connection for the first setup.
2. Double-click **Start Sentinel.bat** in this project folder. It creates a local Python environment, installs the project dependencies, starts the server in a second window, and opens the dashboard when it is ready.
3. Keep the **Sentinel server** window open while using the site. Close it or press Ctrl+C there to stop the server.

The first start can take a few minutes while Python packages install. If setup or startup fails, read the message in the Sentinel server window.

## Quick start on macOS or Linux

Requires **Python 3.11 or newer** (the committed model is pinned to scikit-learn
1.8.0, which needs 3.11+). `run.sh` picks the newest suitable `python3.x` it can
find and tells you what to install if there isn't one — on a Mac,
`brew install python@3.12` works.

From this project folder, run:

```bash
./run.sh
```

Then open **http://127.0.0.1:8000**. The dashboard opens in **Simple** mode
(plain‑language verdicts, AI summaries, big headline numbers); flip the header
toggle to **Analyst** for risk scores, counterfactual evidence, drift PSI and charts.

Containerised, with two workers sharing warm state via Redis:

```bash
docker compose up --scale sentinel=2
```

### Deploy it publicly (free)

| Host | Fit | |
|---|---|---|
| **Hugging Face Spaces** (Docker) | ✅ full app, WebSocket, free forever | `git push` + 8‑line README header |
| **Render** | ✅ full app, WebSocket | New → Blueprint (reads `render.yaml`) |
| **Fly.io** | ✅ full app, WebSocket | `fly launch --copy-config --now` |
| **Vercel** | ⚠️ lite mode — poll‑driven, no live push | `vercel` (reads `vercel.json`) |

A pre‑trained model is committed so every target boots instantly. Full steps in
[docs/DEPLOY.md](docs/DEPLOY.md). `SENTINEL_SERVERLESS=1` switches the app to
poll mode (`POST /tick`) for request‑scoped hosts.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m sentinel.train                       # -> sentinel/artifacts/model.joblib
python -m sentinel.eval                        # honest evaluation on an unseen world
python -m sentinel.audit                       # latency + fairness assurance
python -m sentinel.datasets.fetch ulb          # download 284,807 real transactions
uvicorn sentinel.main:app --port 8000
pytest                                         # 36 tests
```

---

## Dashboard

- Live decision stream, colour‑coded, with risk bar.
- **Policy A/B strip** — rules‑only vs model‑only vs Sentinel, detection & FP,
  updating live on the same traffic.
- Click a transaction → **AI summary** (an analyst‑style narrative composed from
  the model's own feature probes + rules — no LLM, no cost), a **grouped
  counterfactual path** with the calibrated probability before and after the hypothetical
  reference changes, and single-feature probe bars. The path is a model comparison, not
  a causal explanation or customer action recommendation.
  **Confirm fraud / Mark legitimate** buttons that feed the feedback loop.
- **🔬 What‑if explorer** — an interactive counterfactual: drag the amount
  multiplier, toggle "unrecognised device / new payee / overnight / …", and the
  decision + risk + narrative re‑score live (`POST /whatif`). Shows what would
  have flipped the outcome.
- Header shows OOT metrics, calibrated Brier, the state backend, and a **live
  PSI drift indicator** (flips to `alert` during an attack wave).
- **⚡ / 🎭 attack buttons** — inject a full episode (4 classic + 3 adversarial
  playbooks) against a random real customer and watch Sentinel respond.

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /score` | score one transaction → decision, features, and `counterfactual` evidence for analyst-review cases |
| `GET /metrics` | detection / FP / exposure counters, per‑scenario recall |
| `GET /drift` | PSI drift status |
| `GET /metrics` → `policy_comparison` | live rules‑only vs model‑only vs full |
| `POST /whatif` | `{case_id, overrides}` — re‑score a past case with changed features → counterfactual action + narrative |
| `POST /feedback` | `{cust_id, ts, amount, label, kind}` — analyst/chargeback label |
| `GET /cases?only=alerts` | recent decisions (each carries `summary` + `shadows`) |
| `POST /simulator/inject/{scenario}` | play an attack episode |
| `WS /ws/stream` | live push of decisions + metrics + drift |

```bash
curl -X POST localhost:8000/score -H 'content-type: application/json' -d '{
  "cust_id":"C00042","amount":4200,"mcc":"wire_transfer","channel":"transfer",
  "merchant_id":"acct_991","beneficiary":"acct_991","country":"UA",
  "lat":50.45,"lon":30.52,"device_id":"dev-new"}'
```

---

## Layout

```
sentinel/
  config.py         every threshold / weight / cost, one file
  datasets/         synthetic.py (7 playbooks) · csv_adapter.py (ULB/IEEE/Sparkov)
                    fetch.py (download real data) · load_events()
  features.py       ProfileState + compute_features() + sequence head  (shared train ↔ serve)
  entities.py       EntityRegistry — decayed fraud rates + graph fan-in/out
  model.py          HGB + isotonic calibration + cost threshold + IsolationForest
  explain.py        grouped counterfactual reference search
  rules.py          deterministic tripwires
  decision.py       score + rules → action + reasons + alert + shadow policies
  drift.py          PSI monitor
  feedback.py       analyst/chargeback store + label-maturity weights
  state.py          snapshot persistence (disk | Redis)
  engine.py         the real-time pipeline + metrics + policy A/B
  simulator.py      async ambient traffic (simulated clock) + attack injection
  audit.py          latency percentiles + fairness / disparate-impact
  train.py / eval.py
  main.py           FastAPI + dashboard
tests/              36 tests — features, entities, sequence, rules, model, drift,
                    feedback, state, policy A/B, csv adapters, fetch, audit,
                    full pipeline, API
Dockerfile · docker-compose.yml   two workers + Redis
docs/             DATA.md · REAL_DATA_VALIDATION.md · MODEL_CARD.md
```

---

## Next

- Sequence head (GRU/Transformer over each card's last N transactions).
- Graph neural net on the shared‑entity graph for ring detection at scale.
- Nightly scheduled retrain triggered by the PSI `alert` state.
- Per‑segment cost thresholds (a blocked $3 card‑test ≠ a blocked $3 000 payroll run).

*Built for a hackathon. Defensive security use only.*
