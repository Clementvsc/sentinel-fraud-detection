"""Honest evaluation on a fresh, unseen synthetic world.

    python -m sentinel.eval

Runs every transaction of a held-out world (different seed) through the *full*
production pipeline — calibrated model + anomaly head + rules + decision engine —
and reports:

    * decision mix, detection rate, false-positive rate, precision/recall
    * per-scenario recall, including the three adversarial evasion playbooks
    * probability calibration (reliability table, ECE, Brier)
    * drift/PSI of this world's scores vs. the training reference
"""
from __future__ import annotations

import numpy as np

from . import config
from .datasets import SyntheticSource
from .drift import psi
from .engine import Engine
from .model import FraudModel


def _calibration(proba: np.ndarray, y: np.ndarray, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for i in range(bins):
        m = (proba >= edges[i]) & (proba < edges[i + 1] if i < bins - 1 else proba <= 1.0)
        if not m.any():
            continue
        conf, acc, n = proba[m].mean(), y[m].mean(), int(m.sum())
        ece += n / len(y) * abs(acc - conf)
        rows.append((edges[i], edges[i + 1], n, conf, acc))
    return rows, ece


def main() -> None:
    if not FraudModel.exists():
        raise SystemExit("no model artifact — run `python -m sentinel.train` first")
    model = FraudModel.load()
    engine = Engine(model, autosnapshot=False)
    # unseen world: different seed, its own customers
    src = SyntheticSource(110, 35, seed=config.TRAIN_SEED + 999)
    engine.customers = {c.cust_id: c for c in src.customers}
    for c in src.customers:
        from .features import ProfileState
        engine.profiles[c.cust_id] = ProfileState(
            c.cust_id, c.home_country, c.home_lat, c.home_lon, c.account_open)

    events = src.events()
    n_txn = sum(e["type"] == "txn" for e in events)
    print(f"  scoring {n_txn:,} transactions through the full pipeline ...")
    probas, risks, ys, scen_hit, scen_tot = [], [], [], {}, {}
    done = 0
    for ev in events:
        if ev["type"] == "login":
            ps = engine.profiles.get(ev["cust_id"])
            if ps:
                ps.add_login(ev["ts"], ev["success"], ev.get("device_id", ""))
            continue
        case = engine.process(ev)
        done += 1
        if done % 2000 == 0:
            print(f"    {done:,}/{n_txn:,}")
        probas.append(case["fraud_proba"])
        risks.append(case["risk"])
        ys.append(case["label"])
        if case["label"] == 1:
            s = case["scenario"]
            scen_tot[s] = scen_tot.get(s, 0) + 1
            scen_hit[s] = scen_hit.get(s, 0) + int(case["action"] in ("BLOCK", "CHALLENGE"))

    m = engine.metrics_snapshot()
    probas, ys = np.asarray(probas), np.asarray(ys)

    print("\n  policy comparison (rules-only vs model-only vs Sentinel):")
    print(f"    {'policy':<12} {'detection':>10} {'FP rate':>10}")
    for name in ("rules_only", "model_only", "full"):
        pc = m["policy_comparison"][name]
        print(f"    {name:<12} {pc['detection_rate']*100:>9.1f}% {pc['false_positive_rate']*100:>9.3f}%")

    print("\n=== Sentinel evaluation — fresh held-out world ===")
    print(f"  transactions      : {m['processed']:,}")
    print(f"  fraud             : {m['fraud_total']}  ({m['fraud_total']/m['processed']*100:.2f}%)")
    print(f"  decision mix      : {m['by_action']}")
    print(f"  detection rate    : {m['detection_rate']*100:.1f}%  "
          f"(BLOCK or CHALLENGE on labelled fraud)")
    print(f"  false-positive rate: {m['false_positive_rate']*100:.3f}%  "
          f"({m['false_positives']} legit txns blocked)")
    print(f"  exposure prevented: ₹{m['amount_saved']:,.0f}")

    print("\n  per-scenario recall:")
    for s in sorted(scen_tot):
        tag = "  (adversarial)" if s in {"amount_just_under", "slow_drip", "geo_consistent_ato"} else ""
        print(f"    {s:20s} {scen_hit[s]:3d}/{scen_tot[s]:<3d}  "
              f"{scen_hit[s]/scen_tot[s]*100:5.1f}%{tag}")

    rows, ece = _calibration(probas, ys)
    brier = float(np.mean((probas - ys) ** 2))
    print(f"\n  calibration: ECE={ece:.4f}  Brier={brier:.5f}")
    print("    bin            n     predicted   actual")
    for lo, hi, n, conf, acc in rows:
        print(f"    [{lo:.1f},{hi:.1f})  {n:6d}   {conf:8.3f}   {acc:7.3f}")

    ref = model.meta.get("ref_scores", [])
    print(f"\n  PSI (risk blend) vs training reference: {psi(ref, list(risks)):.4f}")


if __name__ == "__main__":
    main()
