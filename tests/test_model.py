import numpy as np
from sklearn.metrics import brier_score_loss

from sentinel.features import FEATURE_COLUMNS


def test_temporal_holdout_contains_fraud_and_scores_well(trained_model):
    _model, meta = trained_model
    assert meta["n_fraud"] > 20
    assert meta["roc_auc"] > 0.9
    assert 0.0 < meta["cost_threshold"] < 1.0
    assert "time-ordered" in meta["split"]


def test_calibration_improves_brier(trained_model):
    _model, meta = trained_model
    assert meta["brier_calibrated"] <= meta["brier_raw"] + 1e-9


def test_score_blend_and_ranges(trained_model):
    model, _ = trained_model
    feat = {c: 0.0 for c in FEATURE_COLUMNS}
    feat["amount"] = 50.0
    s = model.score(feat, explain=True)
    assert 0.0 <= s.risk <= 1.0 and 0.0 <= s.fraud_proba <= 1.0
    names = [n for n, _v, _c in s.top_features]
    assert names and all(n in FEATURE_COLUMNS for n in names)


def test_explainer_mode_is_counterfactual(trained_model):
    model, _ = trained_model
    assert model.explainer.mode == "counterfactual"
