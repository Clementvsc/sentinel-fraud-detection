"""Central configuration for Sentinel.

Every tunable lives here so the pipeline can be reasoned about at a glance and
reviewers can see exactly which knobs drive a BLOCK / CHALLENGE / ALLOW decision.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
SNAPSHOT_PATH = ARTIFACT_DIR / "state.snapshot"
FEEDBACK_PATH = ARTIFACT_DIR / "feedback.jsonl"
QUEUE_PATH = ARTIFACT_DIR / "queue.json"
FRONTEND_DIR = ROOT / "frontend"

# --------------------------------------------------------------------------- #
# Decision thresholds  (risk score is 0..1, built from a *calibrated* proba)
# --------------------------------------------------------------------------- #
RISK_BLOCK = 0.90       # >= this OR a hard rule  -> transaction is stopped
RISK_CHALLENGE = 0.55   # >= this                 -> step-up auth (OTP / passkey)
RISK_REVIEW = 0.28      # >= this                 -> allowed but queued for an analyst

# Blend of the two model heads that produces the final risk score.
W_SUPERVISED = 0.75     # calibrated gradient-boosted fraud classifier
W_ANOMALY = 0.25        # isolation-forest novelty detector

# Cost-sensitive operating point. The classifier threshold used for the
# supervised head's *label* is chosen to minimise expected cost on validation.
COST_FALSE_NEGATIVE = 1.0    # relative cost of letting a fraud txn through (per $1)
COST_FALSE_POSITIVE = 0.04   # relative cost of blocking a legit txn (friction/churn)
#   -> a blocked legit txn "costs" as much as missing ~$0.04 of fraud per $1.

# --------------------------------------------------------------------------- #
# Rule-engine constants (deterministic tripwires)
# --------------------------------------------------------------------------- #
IMPOSSIBLE_TRAVEL_KMH = 900.0
IMPOSSIBLE_TRAVEL_MIN_KM = 200.0
CARD_TESTING_TXNS_5M = 6
VELOCITY_TXNS_1H = 12
ATO_FAILED_LOGINS = 5
HIGH_RISK_MCC = {"crypto", "gift_card", "wire_transfer"}
ENTITY_FRAUD_RATE_BLOCK = 0.22   # a payee/device seen >=22% on fraud (~70x base rate) is blocked
RING_SIZE_CHALLENGE = 4          # distinct victims sharing this device/beneficiary

# --------------------------------------------------------------------------- #
# Entity aggregation  (Bayesian-smoothed historical fraud rates)
# --------------------------------------------------------------------------- #
ENTITY_PRIOR_STRENGTH = 20.0     # pseudo-counts pulling a fresh entity to base rate
ENTITY_HALFLIFE_DAYS = 30.0      # exponential decay on entity fraud statistics

# --------------------------------------------------------------------------- #
# Drift monitoring (Population Stability Index)
# --------------------------------------------------------------------------- #
PSI_WINDOW = 2000                # rolling window of ALLOWED-traffic risk scores
PSI_WATCH = 0.18
PSI_ALERT = 0.35
DRIFT_BASELINE_AFTER = 2500      # re-baseline the reference on real live traffic
                                 # after this many transactions (production drift
                                 # monitors baseline on post-go-live traffic, not
                                 # the training set)

# --------------------------------------------------------------------------- #
# Synthetic-data / training parameters
# --------------------------------------------------------------------------- #
TRAIN_CUSTOMERS = 400
TRAIN_DAYS = 90
TRAIN_SEED = 42
FRAUD_VICTIM_RATE = 0.22         # share of customers with >=1 fraud episode
DATA_SOURCE = os.getenv("SENTINEL_DATA", "synthetic")   # or a path to a CSV
CSV_SCHEMA = os.getenv("SENTINEL_CSV_SCHEMA", "sparkov")  # sparkov | ulb | ieee

# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
RUNTIME_CUSTOMERS = 240
RUNTIME_SEED = 7
SEED_HISTORY_DAYS = 90
AMBIENT_TXNS_PER_SEC = 2.0
AMBIENT_FRAUD_RATE = 0.004
CASE_BUFFER = 500
SNAPSHOT_EVERY = 1000          # persist state every N processed transactions
REDIS_URL = os.getenv("SENTINEL_REDIS_URL", "")   # empty -> in-memory + disk snapshot

# Serverless / no-background-loop hosting (e.g. Vercel functions): the ambient
# stream is driven by the client polling POST /tick instead of an asyncio task,
# and attack injection is processed synchronously.
SERVERLESS = os.getenv("SENTINEL_SERVERLESS", "").lower() in ("1", "true", "yes")
