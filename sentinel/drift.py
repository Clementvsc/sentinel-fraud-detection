"""Population Stability Index drift monitoring.

Compares the distribution of live risk scores against a reference distribution
(the held-out test set at training time, or, once enough real traffic has
flowed, the live population itself).

Two design choices keep this from crying wolf:

  * only **allowed** traffic is measured — a burst of injected fraud is a
    *different* alarm, not model drift. Drift asks "does my normal population
    still look normal?".
  * "alert" requires the shift to be **sustained** (an EWMA of PSI), so a
    transient blip during an attack wave doesn't flip the badge.

    smoothed PSI < 0.15   stable
    0.15 - 0.35           watch  — investigate
    > 0.35               alert  — population has shifted, consider retraining

Robustness fixes:

  * The reference is normalised on the way in. It may be a ready-made
    10-bin histogram, a list of raw scores (converted to a histogram), or
    empty. Previously raw scores were used as if they were histogram bins,
    and an empty list fell back to a FLAT histogram, which makes almost any
    real traffic look like a huge shift (PSI of 3-4).
  * With no usable reference the monitor reports "warming" and baselines on
    the first real allowed traffic instead of comparing against a made-up one.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .config import DRIFT_BASELINE_AFTER, PSI_ALERT, PSI_WATCH, PSI_WINDOW

_FLOOR = 0.005      # standard PSI practice: floor each bin at 0.5% so the log
                    # ratio stays finite when a reference bin is (near) empty
PSI_RANGE_HI = 0.5  # allowed traffic is all below this; binning over [0, 0.5]
                    # instead of [0, 1] gives real resolution and avoids the
                    # empty-bin blow-up on a low-entropy score distribution
N_BINS = 10


def score_histogram(scores, bins: int = N_BINS) -> list[float]:
    h, _ = np.histogram(np.clip(scores, 0, PSI_RANGE_HI), bins=bins, range=(0, PSI_RANGE_HI))
    h = h.astype(float) + 1e-6
    return (h / h.sum()).tolist()


def _as_histogram(reference) -> list[float] | None:
    """Turn whatever was supplied as a reference into a normalised 10-bin
    histogram, or None if there is nothing usable."""
    if reference is None:
        return None
    arr = np.asarray(list(reference), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    looks_like_histogram = (
        arr.size == N_BINS and arr.min() >= 0.0 and abs(arr.sum() - 1.0) < 1e-3
    )
    if looks_like_histogram:
        return arr.tolist()
    return score_histogram(arr)          # raw scores -> histogram


def psi(reference: list[float], observed: list[float], bins: int | None = None) -> float:
    ref = np.asarray(reference, dtype=float)
    if len(observed) == 0:
        return 0.0
    bins = bins or len(ref)
    obs_hist, _ = np.histogram(np.clip(observed, 0, PSI_RANGE_HI), bins=bins, range=(0, PSI_RANGE_HI))
    obs = np.maximum(obs_hist.astype(float) / max(obs_hist.sum(), 1), _FLOOR)
    ref = np.maximum(ref / max(ref.sum(), 1e-9), _FLOOR)
    obs /= obs.sum()
    ref /= ref.sum()
    return float(np.sum((obs - ref) * np.log(obs / ref)))


class DriftMonitor:
    MIN_SAMPLES = 400        # PSI is a large-sample statistic

    def __init__(self, reference_scores=None):
        self.reference: list[float] | None = _as_histogram(reference_scores)
        self.window: deque[float] = deque(maxlen=PSI_WINDOW)
        self.count = 0
        self._ewma: float | None = None       # smoothed PSI
        self._baselined = False

    def observe(self, risk: float, allowed: bool = True) -> None:
        self.count += 1
        if not allowed:                       # attack traffic ≠ drift
            return
        self.window.append(float(risk))

        # Re-baseline the reference on real allowed traffic once enough has
        # flowed — or immediately (after a minimal window) if we never had a
        # usable reference to begin with.
        due = self.reference is None or self.count >= DRIFT_BASELINE_AFTER
        if (not self._baselined and due
                and len(self.window) >= min(PSI_WINDOW, 800)):
            self.reference = score_histogram(list(self.window))
            self._ewma = None
            self._baselined = True

        if self.reference is not None and len(self.window) >= self.MIN_SAMPLES:
            raw = psi(self.reference, list(self.window))
            self._ewma = raw if self._ewma is None else 0.9 * self._ewma + 0.1 * raw

    def status(self) -> dict:
        n = len(self.window)
        ref_bins = [round(x, 4) for x in self.reference] if self.reference else []
        live_bins = [round(x, 4) for x in score_histogram(list(self.window))] if n else []
        common = {
            "window": n,
            "window_capacity": PSI_WINDOW,
            "reference_bins": ref_bins,
            "live_bins": live_bins,      # compare with reference_bins to see WHERE it shifted
            "baselined": self._baselined,
        }
        if self.reference is None or n < self.MIN_SAMPLES or self._ewma is None:
            return {"psi": 0.0, "psi_raw": 0.0, "state": "warming", **common}
        raw = psi(self.reference, list(self.window))
        v = self._ewma
        state = "alert" if v > PSI_ALERT else "watch" if v > PSI_WATCH else "stable"
        return {"psi": round(v, 4), "psi_raw": round(raw, 4), "state": state, **common}
