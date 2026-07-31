r"""Is a drift alarm worth more than a cron job?

The published benchmarks evaluate detectors against DEGRADATION: does
an alarm coincide with a drop in performance? That leaves the question
a practitioner actually faces unanswered, because it never tests the
ACTION the alarm recommends.

Retraining helps for two quite different reasons, and comparing
"monitored" against "unmonitored" confounds them:

  1. retraining on fresher data helps, whenever you do it
  2. retraining AT THE RIGHT MOMENT helps

Only the second is attributable to the detector. So the three policies
here are matched on retraining FREQUENCY — identical compute, identical
retrain rule, differing only in WHEN they fire:

  A  never       train once, never update
  B  scheduled   retrain every k windows
  C  triggered   retrain when a detector fires, its threshold calibrated
                 so that it fires at the same average rate as B

If C beats B, the signal carries timing information. If C ties B, an
entire category of monitoring infrastructure is worth no more than a
schedule.

Evaluation is prequential: each window is scored by the model BEFORE
any retraining that window would trigger, so no policy is ever tested
on data it has already seen.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 5000-row windows exceed scipy's exact KS threshold; the asymptotic
# result is what we want and the warning is pure noise.
warnings.filterwarnings("ignore", message=".*ks_2samp.*")


# ----------------------------------------------------------- detectors
def ks_statistic(ref: np.ndarray, cur: np.ndarray) -> float:
    """Largest per-feature KS statistic between reference and current.
    Standard practice: a feature-distribution detector, blind to the
    label relationship by construction."""
    return max(stats.ks_2samp(ref[:, j], cur[:, j]).statistic
               for j in range(ref.shape[1]))


def psi_statistic(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Largest per-feature Population Stability Index."""
    out = 0.0
    for j in range(ref.shape[1]):
        edges = np.quantile(ref[:, j], np.linspace(0, 1, bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref[:, j], edges)[0] / len(ref) + 1e-6
        c = np.histogram(cur[:, j], edges)[0] / len(cur) + 1e-6
        out = max(out, float(np.sum((c - r) * np.log(c / r))))
    return out


DETECTORS = {"ks": ks_statistic, "psi": psi_statistic}


# ------------------------------------------------------------ policies
def _fit(X, y):
    return LogisticRegression(max_iter=400).fit(X, y)


def _score(model, X, y):
    """AUC, not accuracy.

    With a 74% base rate, accuracy cannot separate "the model has no
    signal" from "the model has signal and it is stable" — and that
    distinction is the whole experiment. AUC is invariant to the base
    rate and reads 0.5 when a model is worthless, which makes a null
    run announce itself.
    """
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, model.predict_proba(X)[:, 1]))


def persistence_auc(windows):
    """AUC of the dumbest possible predictor: last observed label.

    This exists because the usability gate turned out to correlate with
    autocorrelation at r = +0.78 across the streams tested. When labels
    are sticky, a stale model falls out of step with the CURRENT REGIME
    and appears to decay, and any detector that fires on distributional
    change will track that regime. Apparent drift value would then be
    momentum, not concept.

    If this reference approaches what the retraining policies achieve,
    the stream is not testing what the experiment claims to test.
    """
    scores, labels = [], []
    prev = windows[0][1][-1]
    for X, y in windows[1:]:
        for yi in y:
            scores.append(float(prev))
            labels.append(int(yi))
            prev = yi
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def majority_baseline(windows):
    """What always predicting the majority class would score, on
    accuracy. Reported alongside so a model that beats nothing is
    visible immediately."""
    accs = []
    for X, y in windows[1:]:
        p = y.mean()
        accs.append(max(p, 1 - p))
    return float(np.mean(accs))


def run_policy(windows, policy, k=5, detector="ks", threshold=None,
               memory=3):
    """Prequential run. `memory` = how many recent windows a retrain
    uses; identical for B and C so the sample-size effect cancels.

    Returns (mean accuracy over scored windows, number of retrains).
    """
    fn = DETECTORS[detector]
    X0, y0 = windows[0]
    model = _fit(X0, y0)
    ref = X0
    accs, retrains = [], 0

    for t in range(1, len(windows)):
        Xt, yt = windows[t]
        accs.append(_score(model, Xt, yt))       # score BEFORE retraining

        fire = False
        if policy == "never":
            fire = False
        elif policy == "scheduled":
            fire = (t % k == 0)
        elif policy == "triggered":
            fire = fn(ref, Xt) > threshold
        else:
            raise ValueError(policy)

        if fire:
            lo = max(0, t - memory + 1)
            Xr = np.vstack([windows[i][0] for i in range(lo, t + 1)])
            yr = np.concatenate([windows[i][1] for i in range(lo, t + 1)])
            model = _fit(Xr, yr)
            ref = Xt
            retrains += 1

    return float(np.mean(accs)), retrains


def count_fires(windows, detector, threshold):
    """How often would this threshold fire? Mirrors run_policy's
    reference-window logic exactly, but skips model fitting — calibration
    only needs the firing pattern, and fitting made it ~50x slower."""
    fn = DETECTORS[detector]
    ref = windows[0][0]
    n = 0
    for t in range(1, len(windows)):
        if fn(ref, windows[t][0]) > threshold:
            n += 1
            ref = windows[t][0]
    return n


def calibrate_threshold(windows, detector, target_rate, memory=3):
    """Find the detector threshold whose firing rate matches the
    schedule's. Without this the comparison is not a comparison — a
    detector that fires more often would win on retraining frequency
    alone, which is exactly the confound this experiment exists to
    remove."""
    fn = DETECTORS[detector]
    lo, hi = 0.0, 10.0
    for _ in range(24):
        mid = (lo + hi) / 2
        rate = count_fires(windows, detector, mid) / (len(windows) - 1)
        if rate > target_rate:
            lo = mid          # firing too often -> raise the bar
        else:
            hi = mid
    return (lo + hi) / 2
