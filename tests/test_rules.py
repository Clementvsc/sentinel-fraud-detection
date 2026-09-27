from datetime import datetime, timedelta

from sentinel.entities import EntityRegistry
from sentinel.features import ProfileState, compute_features
from sentinel.rules import evaluate_rules, worst_severity


def _ps():
    ps = ProfileState("C1", "US", 40.71, -74.01, datetime(2024, 1, 1))
    base = datetime(2025, 6, 1, 12)
    for i in range(40):
        ps.update({"amount": 40 + (i % 5) * 4, "merchant_id": f"grocery_{i%3}",
                   "country": "US", "device_id": "dev-a", "beneficiary": "",
                   "lat": 40.71, "lon": -74.01, "ts": base - timedelta(days=40 - i)})
    return ps, base


def _txn(ps_now, **kw):
    base = dict(cust_id="C1", ts=ps_now, amount=60.0, mcc="grocery", channel="pos",
               merchant_id="grocery_1", beneficiary="", country="US", city="NYC",
               lat=40.71, lon=-74.01, device_id="dev-a", card_bin="440000")
    base.update(kw)
    return base


def test_impossible_travel_blocks():
    ps, now = _ps()
    ps.update(_txn(now))
    txn = _txn(now + timedelta(minutes=30), country="SG", lat=1.35, lon=103.82,
               merchant_id="x", mcc="electronics")
    feat = compute_features(txn, ps, EntityRegistry(), txn["ts"])
    assert worst_severity(evaluate_rules(feat, txn)) == "block"


def test_card_testing_blocks():
    ps, now = _ps()
    hits = []
    for i in range(7):
        t = now + timedelta(seconds=20 * i)
        txn = _txn(t, amount=1.5, channel="online", merchant_id=f"probe_{i}",
                   device_id="dev-bot", mcc="retail")
        feat = compute_features(txn, ps, EntityRegistry(), t)
        hits = evaluate_rules(feat, txn)
        ps.update(txn)
    assert any(h.code == "card_testing" for h in hits)


def test_known_bad_entity_and_ring_rules():
    ps, now = _ps()
    reg = EntityRegistry()
    for i in range(30):
        reg.observe(_txn(now, merchant_id="badm", beneficiary="mule-1",
                         channel="transfer", cust_id=f"V{i}"), 1, now)
    txn = _txn(now + timedelta(hours=2), merchant_id="badm", beneficiary="mule-1",
               channel="transfer", amount=300)
    feat = compute_features(txn, ps, reg, txn["ts"])
    codes = {h.code for h in evaluate_rules(feat, txn)}
    assert "known_bad_entity" in codes
    assert "fraud_ring" in codes


def test_normal_txn_is_clean():
    ps, now = _ps()
    txn = _txn(now + timedelta(hours=6), amount=44, merchant_id="grocery_1")
    feat = compute_features(txn, ps, EntityRegistry(), txn["ts"])
    assert evaluate_rules(feat, txn) == []
