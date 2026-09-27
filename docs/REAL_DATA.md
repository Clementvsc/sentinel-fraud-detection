# Replaying real transactions live (not just training on them)

`docs/DATA.md` covers training on real datasets offline. This is the
companion piece: streaming **genuine, historical transactions** through the
*live* dashboard — the same `/score` path the simulator uses — so what you
see scrolling by is real merchant/amount/location/fraud-label data instead of
synthetic traffic.

## Fastest path: Sparkov (recommended)

1. Get a free Kaggle account: https://www.kaggle.com/account/login
2. Download **"Credit Card Transactions Fraud Detection Dataset"**:
   https://www.kaggle.com/datasets/kartik2112/fraud-detection
   (grab `fraudTrain.csv`, ~180 MB)
3. Put it at `data/sparkov.csv` relative to the project root (create the
   `data/` folder if it doesn't exist — it's git-ignored already):
   ```bash
   mkdir -p data
   mv ~/Downloads/fraudTrain.csv data/sparkov.csv
   ```
4. With the server running, either:
   - click **"Replay real transactions"** in the dashboard, or
   - call the API directly:
     ```bash
     curl -X POST localhost:8000/replay -H 'content-type: application/json' \
       -d '{"schema_name": "sparkov", "limit": 100}'
     ```

Each call replays the next batch of rows (it remembers where it left off per
dataset) through the exact same scoring, decisioning, and explanation
pipeline as live traffic — real amounts, real merchants, real fraud labels
(`is_fraud` in the source data), broadcast to every connected dashboard.

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
real identity to compare against). Sparkov's rows carry real card IDs,
merchants, cities, and coordinates, so they exercise the full live pipeline
properly.

## Going further: real-time feeds instead of a static file

`docs/DATA.md` lists a few free sandbox APIs (Plaid Sandbox, GoCardless,
Lithic/Marqeta, Stripe test mode) that push transaction-like events over a
webhook or poll. Wiring one in is the same shape as `/replay`: translate its
payload into the canonical event dict and call `engine.process(txn)` (or
just `POST` it to `/score`) as each one arrives.
