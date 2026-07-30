r"""A drift stream where we control whether the drift actually matters.

The whole question Tripwire asks is whether a drift alarm carries
information a fixed schedule does not. To test the machinery for that
we need streams where the right answer is known in advance, which means
separating two things practitioners routinely conflate:

  COVARIATE shift   P(X) moves, P(y|X) is fixed.
                    Detectors fire. The decision boundary is unchanged.
                    Retraining should NOT help, and may hurt by fitting
                    a smaller recent window.

  CONCEPT shift     P(y|X) moves.
                    The boundary itself has moved, so the model is
                    genuinely stale and retraining SHOULD help.
                    Feature-drift detectors may or may not notice.

A detector watching feature distributions can only see the first
directly. If drift alarms are worth more than a cron job, that value
has to come from firing at the right TIMES under concept shift — not
merely from firing at the right RATE.

Both regimes here are generated from an explicit linear model, so the
Bayes-optimal boundary at every timestep is known and the experiment
can be checked against it rather than against another estimate.
"""
from __future__ import annotations

import numpy as np


class DriftStream:
    """A sequence of (X, y) windows with controllable drift.

    covariate_rate : per-window drift of the feature MEANS
    concept_rate   : per-window rotation of the true coefficient vector
    """

    def __init__(self, n_features=8, window_size=500, n_windows=40,
                 covariate_rate=0.0, concept_rate=0.0, noise=0.25, seed=0):
        self.p = n_features
        self.window_size = window_size
        self.n_windows = n_windows
        self.covariate_rate = covariate_rate
        self.concept_rate = concept_rate
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.beta0 = self.rng.normal(size=n_features)
        self.beta0 /= np.linalg.norm(self.beta0)
        # a fixed orthogonal direction to rotate the boundary toward
        d = self.rng.normal(size=n_features)
        d -= d @ self.beta0 * self.beta0
        self.direction = d / np.linalg.norm(d)

    def beta_at(self, t):
        """True coefficients at window t — the boundary the model chases."""
        theta = self.concept_rate * t
        b = np.cos(theta) * self.beta0 + np.sin(theta) * self.direction
        return b / np.linalg.norm(b)

    def mean_at(self, t):
        """Feature means at window t — what a drift detector can see."""
        shift = np.zeros(self.p)
        shift[: self.p // 2] = self.covariate_rate * t
        return shift

    def window(self, t):
        X = self.rng.normal(size=(self.window_size, self.p)) + self.mean_at(t)
        logits = X @ self.beta_at(t) / self.noise
        prob = 1.0 / (1.0 + np.exp(-logits))
        y = (self.rng.random(self.window_size) < prob).astype(int)
        return X, y

    def stream(self):
        for t in range(self.n_windows):
            yield t, *self.window(t)

    def bayes_accuracy(self, t, n=20000):
        """Accuracy of the true model at window t — the ceiling any
        policy could reach, and a check that the stream is what we
        think it is."""
        X = self.rng.normal(size=(n, self.p)) + self.mean_at(t)
        logits = X @ self.beta_at(t) / self.noise
        prob = 1.0 / (1.0 + np.exp(-logits))
        y = (self.rng.random(n) < prob).astype(int)
        return float(((prob > 0.5).astype(int) == y).mean())


REGIMES = {
    # detectors fire, boundary unmoved -> retraining should not help
    "covariate_only": dict(covariate_rate=0.06, concept_rate=0.0),
    # boundary moves, features stay put -> detectors are blind to it
    "concept_only": dict(covariate_rate=0.0, concept_rate=0.045),
    # both at once, the realistic case
    "both": dict(covariate_rate=0.06, concept_rate=0.045),
    # nothing moves -> a control; every policy should tie
    "stationary": dict(covariate_rate=0.0, concept_rate=0.0),
}
