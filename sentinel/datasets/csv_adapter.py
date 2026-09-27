"""Adapters that turn real public fraud datasets into Sentinel events.

    SENTINEL_DATA=/path/creditcard.csv          SENTINEL_CSV_SCHEMA=ulb
    SENTINEL_DATA=/path/fraudTrain.csv          SENTINEL_CSV_SCHEMA=sparkov
    SENTINEL_DATA=/path/ieee_train.csv          SENTINEL_CSV_SCHEMA=ieee
    SENTINEL_DATA=/path/upi_fraud.csv           SENTINEL_CSV_SCHEMA=upi

* **sparkov** — Kaggle "Credit Card Transactions Fraud Detection Dataset".
  Full mapping: has timestamp, card number, merchant, category, amount, geo.
* **ieee**   — Kaggle IEEE-CIS Fraud Detection (transaction file). Partial:
  no geo; (card1, addr1) is used as the customer proxy, DeviceInfo as device,
  P_emaildomain as a beneficiary-like entity.
* **ulb**    — Kaggle ULB "creditcard.csv" (V1..V28 PCA features). No entities,
  so events carry a ``raw_features`` dict and the trainer skips feature
  engineering and learns on the raw columns directly.
* **upi**    — Real Indian UPI/Razorpay-style transaction export (columns:
  timestamp, amount, currency=INR, upi_app, bank, device_fingerprint,
  is_suspicious, fraud_reasons, ...). No geo; device_fingerprint (stable,
  recurring per payer) is used as the customer proxy the same way ieee uses
  (card1, addr1), upi_app as the channel-equivalent merchant, and bank folded
  into the merchant id so a "known bad bank/app pairing" is learnable the same
  way a known-bad merchant is elsewhere. Country fixed to IN.

Uses only the standard library so it streams large files without pandas.
"""
from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

_EPOCH = datetime(2020, 1, 1)


def _f(row: dict, *keys, default=0.0) -> float:
    for k in keys:
        v = row.get(k)
        if v not in (None, "", "NaN", "nan"):
            try:
                return float(v)
            except ValueError:
                pass
    return default


def _hash_bin(value: str) -> str:
    return hashlib.sha1(str(value).encode()).hexdigest()[:6]


def load_csv_events(path: str, schema: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    fn = {"sparkov": _sparkov, "ieee": _ieee, "ulb": _ulb, "upi": _upi}.get(schema)
    if fn is None:
        raise ValueError(f"unknown CSV schema '{schema}' (sparkov | ieee | ulb | upi)")
    with p.open(newline="") as fh:
        events = list(fn(csv.DictReader(fh)))
    events.sort(key=lambda e: e["ts"])
    return events


# --------------------------------------------------------------------------- #
def _sparkov(reader: csv.DictReader) -> Iterator[dict]:
    for r in reader:
        try:
            ts = datetime.strptime(r["trans_date_trans_time"], "%Y-%m-%d %H:%M:%S")
        except (KeyError, ValueError):
            ts = _EPOCH + timedelta(seconds=_f(r, "unix_time"))
        cc = r.get("cc_num", "unknown")
        amt = _f(r, "amt")
        cat = (r.get("category") or "retail").replace("_", "")
        channel = "online" if "net" in cat or "online" in cat else "pos"
        mcc = _norm_mcc(cat)
        yield {
            "type": "txn", "ts": ts, "cust_id": f"cc_{cc}",
            "amount": amt, "mcc": mcc, "channel": channel,
            "merchant_id": (r.get("merchant") or "m").replace("fraud_", ""),
            "beneficiary": "", "country": "US",
            "city": r.get("city", ""),
            "lat": _f(r, "lat"), "lon": _f(r, "long"),
            "device_id": f"card_{cc}", "card_bin": str(cc)[:6],
            "label": int(_f(r, "is_fraud")),
        }


def _ieee(reader: csv.DictReader) -> Iterator[dict]:
    for r in reader:
        ts = _EPOCH + timedelta(seconds=_f(r, "TransactionDT"))
        cust = f"{r.get('card1','?')}_{r.get('addr1','?')}"
        pcd = (r.get("ProductCD") or "W").upper()
        channel = {"W": "pos", "C": "online", "R": "online",
                   "H": "online", "S": "transfer"}.get(pcd, "online")
        yield {
            "type": "txn", "ts": ts, "cust_id": f"ic_{cust}",
            "amount": _f(r, "TransactionAmt"),
            "mcc": f"pcd_{pcd.lower()}", "channel": channel,
            "merchant_id": f"{r.get('card1','?')}_{pcd}",
            "beneficiary": (r.get("P_emaildomain") or "").strip(),
            "country": "US", "city": str(r.get("addr1", "")),
            "lat": 0.0, "lon": 0.0,
            "device_id": (r.get("DeviceInfo") or f"card_{r.get('card1','?')}").strip(),
            "card_bin": str(r.get("card1", "?")),
            "label": int(_f(r, "isFraud")),
        }


def _ulb(reader: csv.DictReader) -> Iterator[dict]:
    for r in reader:
        t = _f(r, "Time")
        raw = {f"V{i}": _f(r, f"V{i}") for i in range(1, 29)}
        raw["Amount"] = _f(r, "Amount")
        yield {
            "type": "txn", "ts": _EPOCH + timedelta(seconds=t),
            "cust_id": "ulb", "amount": _f(r, "Amount"),
            "mcc": "retail", "channel": "online", "merchant_id": "ulb",
            "beneficiary": "", "country": "US", "city": "",
            "lat": 0.0, "lon": 0.0, "device_id": "ulb", "card_bin": "ulb",
            "label": int(_f(r, "Class")), "raw_features": raw,
        }


def _upi(reader: csv.DictReader) -> Iterator[dict]:
    for r in reader:
        try:
            ts = datetime.fromisoformat(r["timestamp"])
        except (KeyError, ValueError):
            continue
        device = (r.get("device_fingerprint") or "unknown").strip()
        bank = (r.get("bank") or "bank").strip()
        app = (r.get("upi_app") or "upi").strip()
        # a failed/declined attempt never redefines "normal" behaviour, same
        # principle ProfileState.update() already applies bank-side; keep
        # only settled transactions so profiles aren't built on noise.
        if (r.get("status") or "").strip().lower() != "success":
            continue
        yield {
            "type": "txn", "ts": ts, "cust_id": f"upi_{device}",
            "amount": _f(r, "amount"),
            "mcc": "upi_transfer", "channel": "transfer",
            "merchant_id": f"{bank}_{app}",
            "beneficiary": "", "country": "IN", "city": "",
            "lat": 0.0, "lon": 0.0,
            "device_id": device, "card_bin": _hash_bin(bank),
            "label": 1 if str(r.get("is_suspicious", "")).strip().lower() == "true" else 0,
        }


_MCC_MAP = {
    "grocerypos": "grocery", "grocerynet": "grocery", "shoppingpos": "retail",
    "shoppingnet": "retail", "gasttransport": "transport", "misc_net": "retail",
    "miscpos": "retail", "entertainment": "entertainment", "food_dining": "restaurant",
    "personal_care": "retail", "health_fitness": "retail", "travel": "travel",
    "kids_pets": "retail", "home": "retail",
}


def _norm_mcc(cat: str) -> str:
    c = cat.lower().replace(" ", "").replace("-", "")
    return _MCC_MAP.get(c, "retail")
