# Training Sentinel on real data

Sentinel ships with a seeded **synthetic** generator so it runs with zero setup.
To train on a real, labelled transaction dataset instead, point it at a CSV:

```bash
SENTINEL_DATA=/path/to/file.csv SENTINEL_CSV_SCHEMA=<schema> python -m sentinel.train
python -m sentinel.eval        # then evaluate
```

All of the datasets below are **free** (Kaggle account required to download, no
payment). The adapter (`sentinel/datasets/csv_adapter.py`) uses only the Python
standard library, so it streams large files without pandas.

| `SENTINEL_CSV_SCHEMA` | Dataset | Fit | Notes |
|---|---|---|---|
| `sparkov` | [Credit Card Transactions Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (`fraudTrain.csv`) | **best** | timestamp, card number, merchant, category, amount, lat/long — maps cleanly to every behavioural + entity feature |
| `ieee` | [IEEE‑CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) (`train_transaction.csv`) | good | rich but obfuscated; no geo. `(card1, addr1)` is the customer proxy, `DeviceInfo` the device, `P_emaildomain` a payee‑like entity |
| `ulb` | [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (`creditcard.csv`) | raw mode | features are PCA components `V1..V28` — Sentinel skips feature engineering and trains the GBM directly on them; behavioural/entity/graph features and most rules are inactive |
| `upi` | Real Indian UPI/Razorpay-style transaction export (timestamp, amount, currency=INR, upi_app, bank, device_fingerprint, status, is_suspicious) | small-data | genuine Indian UPI traffic — real apps (GPay, PhonePe, Amazon Pay, ...) and banks (SBI, HDFC, Axis, ...); `device_fingerprint` is the customer proxy, `upi_app`+`bank` form the merchant id, country fixed to `IN`; only `status == success` rows are loaded (a declined attempt never redefines "normal" behaviour); this specific export is only ~4k rows so treat metrics as directional, not a benchmark |

Example:

```bash
SENTINEL_DATA=~/data/fraudTrain.csv SENTINEL_CSV_SCHEMA=sparkov python -m sentinel.train
```

The time‑ordered train/valid/test split, isotonic calibration, cost‑sensitive
threshold, drift reference and grouped counterfactual explainer all work identically on real data.

---

## Adding another schema

Add a generator function to `csv_adapter.py` that yields the canonical event
dict and register it in the `fn` map in `load_csv_events`:

```python
{
  "type": "txn", "ts": <datetime>, "cust_id": str, "amount": float,
  "mcc": str, "channel": "pos|online|atm|transfer",
  "merchant_id": str, "beneficiary": str, "card_bin": str,
  "country": str, "city": str, "lat": float, "lon": float,
  "device_id": str, "label": 0 | 1,
}
```

Login events (optional, improves the account‑takeover signal):

```python
{"type": "login", "ts": <datetime>, "cust_id": str, "device_id": str, "success": 0 | 1}
```

---

## Live data feeds (all have a free tier / sandbox — none billed)

These are *not required*; they replace the simulator with a real stream.

| Source | Free tier | Shape |
|---|---|---|
| **Plaid Sandbox** | free, instant, no card | `/transactions/sync` — realistic synthetic bank transactions; **Plaid Signal** returns an ACH risk score |
| **GoCardless Bank Account Data** (ex‑Nordigen) | free tier | **real** EU bank transactions with user consent |
| **Lithic** / **Marqeta** sandbox | free sandbox | real‑time **card authorization webhooks** — the network asks you to approve/decline in ~2 s; this is exactly where `POST /score → BLOCK` sits |
| **Stripe** test mode + Radar | free test mode | simulate charges, observe fraud scoring |

To wire one in: translate its webhook/poll payload to the event dict above and
call `engine.process(txn)` (or `POST /score`).
