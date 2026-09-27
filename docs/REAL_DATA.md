# Replaying real transactions live (not just training on them)

`docs/DATA.md` covers training on real datasets offline. This is the
companion piece: streaming **genuine, historical transactions** through the
*live* dashboard — the same `/score` path the simulator uses — so what you
see scrolling by is real merchant/amount/location/fraud-label data instead of
synthetic traffic.

## Fastest path: real Indian UPI data (zero setup, recommended)

A genuine Razorpay/UPI transaction export — real apps (GPay, PhonePe, Amazon
Pay, ...), real banks (SBI, HDFC, Axis, ...), INR amounts — is already
committed at `data/upi.csv`, so there's nothing to download. With the server
running, either:
- click **"Replay 100 real UPI transactions (India)"** in the dashboard, or
- call the API directly:
  ```bash
  curl -X POST localhost:8000/replay -H 'content-type: application/json' \
    -d '{"schema_name": "upi", "limit": 100}'
  ```

`upi` is the `/replay` default schema. Each `device_fingerprint` in the
source data is a stable per-payer identity, so the live engine's per-customer
profile (velocity, new-device, entity fraud-rate history) works exactly as it
does on synthetic traffic — just on real, historical UPI transactions.

## Second dataset: PaySim (download required)

**"Synthetic Financial Datasets For Fraud Detection"** (PaySim) is a
mobile-money simulator with real customer identities (`nameOrig`) and
transaction types (CASH_IN/CASH_OUT/DEBIT/PAYMENT/TRANSFER), useful as a
second, larger source once you've exhausted the UPI export:

1. Get a free Kaggle account: https://www.kaggle.com/account/login
2. Download **"Synthetic Financial Datasets For Fraud Detection"**:
   https://www.kaggle.com/datasets/ealaxi/paysim1
3. Put it at `data/paysim.csv` (create `data/` if needed — it's git-ignored):
   ```bash
   mkdir -p data
   mv ~/Downloads/PS_20174392719_1491204439457_log.csv data/paysim.csv
   ```
4. Click **"Replay 100 real PaySim transactions"**, or:
   ```bash
   curl -X POST localhost:8000/replay -H 'content-type: application/json' \
     -d '{"schema_name": "paysim", "limit": 100}'
   ```

PaySim's raw `amount` is an abstract simulation unit (often in the hundreds
of thousands), not a currency figure, so the adapter log-compresses it into
Sentinel's realistic INR range rather than relabelling it as-is — see
`_paysim_amount_to_inr` in `csv_adapter.py`. Relative ordering (a bigger
PaySim amount stays a bigger INR amount) is preserved.

Sparkov (Kaggle "Credit Card Transactions Fraud Detection Dataset") and IEEE-
CIS also work the same way (`schema_name: "sparkov"` / `"ieee"`) if you want
US-context data instead — see `docs/DATA.md` for their column layouts.

Check what's already downloaded and ready with:
```bash
curl localhost:8000/replay/status
```

## Why not the one-command `fetch ulb` dataset for this?

`python -m sentinel.datasets.fetch ulb` downloads the ULB dataset
automatically (no Kaggle account needed) and is great for **training/eval**
(see `docs/REAL_DATA_VALIDATION.md`), but its rows are anonymized PCA
components with no real customer, merchant, or location identity — every row
shares the same placeholder `cust_id`. That's fine for judging the model in
isolation, but it can't drive the live per-customer engine (impossible
travel, new-device, new-payee, entity fraud-rate history, etc. all need a
real identity to compare against). `upi` and `paysim` rows carry real payer
identities, so they exercise the full live pipeline properly.

## Going further: real-time feeds instead of a static file

`docs/DATA.md` lists a few free sandbox APIs (Plaid Sandbox, GoCardless,
Lithic/Marqeta, Stripe test mode) that push transaction-like events over a
webhook or poll. Wiring one in is the same shape as `/replay`: translate its
payload into the canonical event dict and call `engine.process(txn)` (or
just `POST` it to `/score`) as each one arrives.
