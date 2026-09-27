"""Model evaluation: the metric arithmetic is checked against hand-computed
values and sklearn, so the published report can be trusted independently of
the pipeline that produced it."""
from datetime import datetime, timedelta

import numpy as np
import pytest
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

from sentinel.entities import EntityRegistry
from sentinel.evaluation import bootstrap_ci, calibration_table, confusion, evaluate, rates


def test_confusion_matrix_hand_example():
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    p = [1, 1, 0, 1, 0, 0, 0, 0]
    assert confusion(y, p) == {"tp": 2, "fn": 1, "fp": 1, "tn": 4}


def test_rates_match_sklearn():
    rng = np.random.default_rng(3)
    y = (rng.random(2000) < 0.05).astype(int)
    p = np.where(y == 1, rng.random(2000) < 0.8, rng.random(2000) < 0.02).astype(int)
    r = rates(confusion(y, p))
    assert r["precision"] == pytest.approx(precision_score(y, p))
    assert r["recall"] == pytest.approx(recall_score(y, p))
    assert r["f1"] == pytest.approx(f1_score(y, p))
    assert r["mcc"] == pytest.approx(matthews_corrcoef(y, p))
    cm = confusion(y, p)
    assert r["false_positive_rate"] == pytest.approx(cm["fp"] / (cm["fp"] + cm["tn"]))
    assert r["specificity"] == pytest.approx(1 - r["false_positive_rate"])


def test_undefined_rates_are_none_not_zero():
    """No predicted positives -> precision is undefined, and must not be
    reported as 0% (which would look like a measured, terrible result)."""
    r = rates(confusion([1, 0, 0], [0, 0, 0]))
    assert r["precision"] is None and r["f1"] is None and r["mcc"] is None
    assert r["recall"] == 0.0
    r2 = rates(confusion([0, 0, 0], [0, 1, 0]))
    assert r2["recall"] is None and r2["false_positive_rate"] == pytest.approx(1 / 3)


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    y = (rng.random(3000) < 0.1).astype(int)
    p = np.where(y == 1, rng.random(3000) < 0.7, rng.random(3000) < 0.05).astype(int)
    point = rates(confusion(y, p))["recall"]
    lo, hi = bootstrap_ci(y, p, "recall", n_boot=300)
    assert lo <= point <= hi
    assert hi - lo < 0.15


def test_calibration_table_perfectly_calibrated_input():
    rng = np.random.default_rng(0)
    proba = rng.random(20000)
    y = (rng.random(20000) < proba).astype(int)
    rows, ece, brier = calibration_table(proba, y)
    assert ece < 0.02
    assert sum(r["n"] for r in rows) == 20000
    assert 0.0 < brier < 0.25


# --------------------------------------------------------------------------- #
# the entity fix the evaluation surfaced
# --------------------------------------------------------------------------- #
def _txn(cust, device, card, ts, merchant="m1"):
    return {"cust_id": cust, "device_id": device, "card_bin": card,
            "merchant_id": merchant, "beneficiary": "", "ts": ts}


def _registry_with_background(t0):
    """A registry that has seen realistic legitimate traffic, so the global
    base rate is a plausible ~0.1-1%, not whatever a handful of test rows
    happen to be."""
    reg = EntityRegistry()
    for i in range(3000):
        reg.observe(_txn(f"bg{i % 300}", f"bgdev{i % 300}", f"bgcard{i % 300}",
                         t0, merchant=f"bgm{i % 50}"), 0, t0 - timedelta(hours=1))
    return reg


def test_own_past_fraud_does_not_taint_own_device_and_card():
    """A fraudster once used this customer's own phone and card. Afterwards the
    customer's normal spending must not read as 'device/card has fraud
    history' — that self-contamination is what made past victims' genuine
    transactions get stopped for weeks."""
    t0 = datetime(2025, 3, 1)
    reg = _registry_with_background(t0)
    for i in range(5):
        reg.observe(_txn("victim", "dev-v", "card-v", t0), 1, t0 + timedelta(minutes=i))
    later = t0 + timedelta(days=1)
    f = reg.snapshot_features(_txn("victim", "dev-v", "card-v", later), later)
    assert f["device_fraud_rate"] == pytest.approx(reg.prior, rel=0.2)
    assert f["bin_fraud_rate"] == pytest.approx(reg.prior, rel=0.2)


def test_other_customers_fraud_on_a_shared_device_still_lights_up():
    """The ring signal is preserved: fraud by OTHER customers on the same
    device still raises that device's rate for everyone else."""
    t0 = datetime(2025, 3, 1)
    reg = _registry_with_background(t0)
    for i, cust in enumerate(("v1", "v2", "v3")):
        for k in range(3):
            reg.observe(_txn(cust, "dev-ring", f"card-{cust}", t0), 1, t0 + timedelta(minutes=i * 5 + k))
    later = t0 + timedelta(hours=2)
    f = reg.snapshot_features(_txn("new-victim", "dev-ring", "card-x", later), later)
    assert f["device_fraud_rate"] > 10 * reg.prior
    assert f["device_customer_fanout"] == 3
    # ...and v1's own view still counts v2/v3's fraud on that device
    f1 = reg.snapshot_features(_txn("v1", "dev-ring", "card-v1", later), later)
    assert f1["device_fraud_rate"] > 5 * reg.prior


def test_entity_snapshots_without_per_customer_history_still_load():
    """State snapshots written before by_cust existed must keep working."""
    reg, t0 = EntityRegistry(), datetime(2025, 3, 1)
    reg.observe(_txn("c", "d", "b", t0), 0, t0)
    stat = reg._t["device"]["d"]
    del stat.__dict__["by_cust"]
    later = t0 + timedelta(hours=1)
    reg.snapshot_features(_txn("c", "d", "b", later), later)
    reg.observe(_txn("c", "d", "b", later), 1, later)


# --------------------------------------------------------------------------- #
# end to end on a tiny held-out world, and the endpoint
# --------------------------------------------------------------------------- #
def test_evaluate_produces_a_consistent_report(trained_model):
    model, _ = trained_model
    r = evaluate(model, n_customers=25, days=15, seed=12345, n_boot=50)
    cm = r["confusion_matrix"]
    assert sum(cm.values()) == r["dataset"]["transactions"]
    assert cm["tp"] + cm["fn"] == r["dataset"]["fraud"]
    assert sum(r["decision_mix"].values()) == r["dataset"]["transactions"]
    assert sum(a.get("transactions", 0) for a in r["per_age_bracket"]) == r["dataset"]["transactions"]
    for v in r["metrics"].values():
        assert v is None or -1.0 <= v <= 1.0


def test_evaluate_refuses_the_training_seed(trained_model):
    from sentinel import config
    model, _ = trained_model
    with pytest.raises(ValueError):
        evaluate(model, n_customers=5, days=3, seed=config.TRAIN_SEED)


def test_evaluation_endpoint_serves_the_committed_report():
    from fastapi.testclient import TestClient
    from sentinel.main import app
    from sentinel.evaluation import load_report

    with TestClient(app) as client:
        r = client.get("/evaluation")
    if load_report() is None:
        assert r.status_code == 404
    else:
        assert r.status_code == 200
        j = r.json()
        assert {"confusion_matrix", "metrics", "threshold_free", "per_age_bracket"} <= set(j)
