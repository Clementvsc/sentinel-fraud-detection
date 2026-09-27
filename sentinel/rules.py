"""Deterministic tripwires.

The ML score is the brain; these rules are the reflexes. They exist so a
blatant attack is stopped instantly and *explainably*, even if the model is
uncertain, and so a compliance officer can point at a written policy.

Each rule returns a ``RuleHit`` with a severity that the decision engine maps
to an action:  block > challenge > flag.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import (
    ATO_FAILED_LOGINS,
    CARD_TESTING_TXNS_5M,
    ENTITY_FRAUD_RATE_BLOCK,
    HIGH_RISK_MCC,
    RING_SIZE_CHALLENGE,
    VELOCITY_TXNS_1H,
)


@dataclass(frozen=True)
class RuleHit:
    code: str
    severity: str      # "block" | "challenge" | "flag"
    message: str


def evaluate_rules(feat: dict, txn: dict) -> list[RuleHit]:
    hits: list[RuleHit] = []
    f = feat  # shorthand

    if f["impossible_travel"] >= 1.0:
        hits.append(RuleHit(
            "impossible_travel", "block",
            f"Impossible travel: {f['dist_from_last_km']:.0f} km in "
            f"{f['secs_since_last']/3600:.1f} h (~{f['speed_kmh']:.0f} km/h) "
            f"since the last transaction",
        ))

    if f["txn_count_5m"] >= CARD_TESTING_TXNS_5M:
        hits.append(RuleHit(
            "card_testing", "block",
            f"Card-testing pattern: {int(f['txn_count_5m'])} transactions in 5 minutes",
        ))

    if f["failed_logins_1h"] >= ATO_FAILED_LOGINS and f["amount_to_max"] > 1.0:
        hits.append(RuleHit(
            "ato_login_then_spend", "block",
            f"High-value {txn['channel']} transaction right after "
            f"{int(f['failed_logins_1h'])} failed logins — possible account takeover",
        ))

    if f["new_country"] >= 1.0 and f["amount_z"] > 4.0:
        hits.append(RuleHit(
            "new_country_large_amount", "block",
            f"First-ever transaction in {txn['country']} at "
            f"{f['amount_z']:.1f}σ above this customer's normal spend",
        ))

    if (f["new_device"] >= 1.0 and f["channel_online"] >= 1.0
            and f["amount"] > 5000 and f["failed_logins_1h"] >= 1.0):
        hits.append(RuleHit(
            "new_device_online_spend", "challenge",
            "Large online purchase from an unrecognised device after a failed login",
        ))

    if f["high_risk_mcc"] >= 1.0 and f["new_merchant"] >= 1.0 and f["amount_z"] > 2.0:
        hits.append(RuleHit(
            "high_risk_new_merchant", "challenge",
            f"Large payment to a new {txn['mcc'].replace('_', ' ')} merchant "
            f"({f['amount_z']:.1f}σ above normal)",
        ))

    if f["txn_count_1h"] >= VELOCITY_TXNS_1H:
        hits.append(RuleHit(
            "velocity_1h", "challenge",
            f"Unusual velocity: {int(f['txn_count_1h'])} transactions in the past hour",
        ))

    if f.get("entity_max_fraud_rate", 0.0) >= ENTITY_FRAUD_RATE_BLOCK:
        hits.append(RuleHit(
            "known_bad_entity", "block",
            f"Merchant / device / payee on this transaction has a "
            f"{f['entity_max_fraud_rate']*100:.0f}% historical fraud rate",
        ))

    if f.get("ring_size", 0.0) >= RING_SIZE_CHALLENGE:
        hits.append(RuleHit(
            "fraud_ring", "challenge",
            f"Device or beneficiary is shared across {int(f['ring_size'])} "
            f"different customers — likely a fraud ring / mule account",
        ))

    if (f.get("new_beneficiary", 0.0) >= 1.0 and f["channel_transfer"] >= 1.0
            and f["amount_z"] > 1.5):
        hits.append(RuleHit(
            "new_payee_large_transfer", "flag",
            "Large transfer to a payee this customer has never sent to before",
        ))

    if (f["amount_z"] > 4.0 and f["amount"] > 1500
            and (f["new_merchant"] >= 1.0 or f["is_foreign"] >= 1.0 or f["high_risk_mcc"] >= 1.0)
            and not any(h.code == "new_country_large_amount" for h in hits)):
        hits.append(RuleHit(
            "amount_spike", "flag",
            f"Amount is {f['amount_z']:.1f}σ above this customer's usual spend, "
            f"at a new/foreign/high-risk merchant",
        ))

    if f["is_foreign"] >= 1.0 and f["new_country"] >= 1.0 and f["channel_pos"] >= 1.0:
        hits.append(RuleHit(
            "foreign_pos_first_visit", "flag",
            f"In-person spend in {txn['country']} for the first time",
        ))

    return hits


def worst_severity(hits: list[RuleHit]) -> str | None:
    order = {"flag": 1, "challenge": 2, "block": 3}
    if not hits:
        return None
    return max(hits, key=lambda h: order[h.severity]).severity
