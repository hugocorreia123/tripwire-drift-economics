# Tripwire — Findings

## Is a drift alarm worth more than a cron job?

Hugo Correia · July 2026

---

## Abstract

Production ML teams monitor feature distributions and retrain when a detector
fires. The literature evaluates those detectors against **degradation** — does an
alarm coincide with a drop in performance — which leaves the question a
practitioner actually faces unanswered, because it never tests the **action** the
alarm recommends.

Retraining helps for two separable reasons: fresher data helps whenever you do
it, and retraining *at the right moment* helps. Only the second is attributable
to the detector. This study separates them by matching three policies on
retraining **frequency**: never, a fixed schedule, and a detector calibrated to
fire at the schedule's rate.

**On four synthetic regimes with known ground truth, the drift signal adds
nothing** — the schedule wins or ties everywhere, and loses to nothing. Where
concept drift is invisible to a feature detector, the detector is significantly
**worse** than a schedule at the same cost.

**On real data the answer was initially unclear, then resolved against the
signal.** Two datasets suggested a PSI-based detector might help (18/18
configurations positive, mean +0.0296). That pattern was pre-registered as a
hypothesis and tested on two datasets held out from its discovery. **It did not
replicate**: 4/12 positive, mean −0.0023, refuting the hypothesis under the
decision rule fixed in advance.

A second finding emerged from the gating: **of eleven streams tested, only four
degraded a stale model at all**, and the streams that did degrade correlate with
label autocorrelation at **r = +0.78**. Most standard drift benchmarks cannot
test drift-response questions, and the ones that can may be measuring momentum.

---

## 1. The question

The standard claim for drift monitoring is that it tells you *when* to retrain.
That claim is rarely tested, because the usual evaluation compares a monitored
system against an unmonitored one — which confounds two things:

1. retraining on fresher data helps, whenever you do it
2. retraining at the *right moment* helps

Only the second belongs to the detector. So:

> **Does a drift alarm carry timing information that a schedule of equal
> frequency does not?**

---

## 2. Design

Three policies, **matched on retraining frequency**, identical retrain rule,
differing only in *when* they fire:

| | Policy |
|---|---|
| **A** | never retrain |
| **B** | retrain every *k* windows |
| **C** | retrain when a detector fires, threshold calibrated so its firing rate equals B's |

The frequency matching is the entire design. Without it a detector that fires
more often wins on compute rather than on information.

Evaluation is **prequential**: each window is scored by the model *before* any
retraining that window would trigger, so no policy is tested on data it has
seen. Scoring is **AUC**, not accuracy — with base rates as high as 0.88 in
these streams, accuracy cannot distinguish "no signal" from "signal present and
stable", and that distinction is the whole experiment.

---

## 3. Synthetic validation

Ground truth is known by construction, separating two things practitioners
conflate:

- **Covariate shift** — P(X) moves, P(y|X) fixed. Detectors fire; the boundary
  has not moved.
- **Concept shift** — P(y|X) moves. The model is genuinely stale; feature
  detectors may be blind to it.

Bayes accuracy is held stable across each stream, so any degradation is model
staleness rather than the task getting harder.

| Regime | A never | B schedule | C detector | C − B |
|---|---|---|---|---|
| stationary | 0.867 | 0.870 | 0.870 | −0.0003 |
| covariate only | 0.887 | 0.890 | 0.890 | −0.0002 |
| **concept only** | 0.685 | 0.858 | 0.833 | **−0.0248** ✱ |
| both | 0.668 | 0.870 | 0.869 | −0.0010 |
| coupled | 0.539 | 0.931 | 0.930 | −0.0015 |

**The detector never wins.** Where it matters most — concept drift invisible to
a feature detector — it loses significantly. The mechanism is visible: with
features stationary, the detector fires on sampling noise, at random times.
Random timing at a fixed rate is worse than regular timing, because regular
spacing bounds the maximum staleness while random spacing permits long gaps.

The `coupled` regime was built specifically to test whether a detector helps
when concept drift is *driven by* covariate drift — the situation in several
real datasets. It does not.

---

## 4. Which real streams can test this at all

A stream where a stale model does not decay cannot test the hypothesis: nothing
can help, and every policy ties for an uninteresting reason. Two gates were
applied before any comparison:

- **degradation** — retraining must gain > 0.02 AUC
- **persistence** — accuracy of "predict the previous label" must be ≤ 0.90

| Stream | persistence | retrain gain | verdict |
|---|---|---|---|
| nomao | 0.948 | +0.1855 | excluded — too autocorrelated |
| gas-drift | 0.863 | +0.0437 | **usable** |
| electricity | 0.853 | +0.1956 | **usable** |
| bank-marketing | 0.843 | +0.0693 | **usable** |
| covertype | 0.798 | +0.1324 | **usable** |
| shuttle | 0.664 | −0.0002 | excluded — no degradation |
| airlines | 0.546 | −0.0047 | excluded — no degradation |
| poker | 0.498 | +0.0030 | excluded — no degradation |
| NYC taxi 2024–25 | — | +0.0003 | excluded — no degradation |
| NYC taxi, COVID window | — | +0.0034 | excluded — no degradation |

**Only four of eleven streams qualify.** The taxi result is worth stating
separately: across the COVID collapse, trip volume fell **97%** — 6.9M journeys
in November 2019 to 238k in April 2020 — and the tip-prediction boundary barely
moved. Enormous covariate shift, essentially no concept drift. Every detector
in production would have fired; nothing needed doing.

### 4.1 The gate correlates with autocorrelation

| | |
|---|---|
| correlation between persistence and retraining gain | **r = +0.78** |
| every stream that passed | persistence 0.80–0.87 |
| every stream excluded for no degradation | persistence below 0.67 |

This is a confound, and it was found at the gating stage, before any
confirmatory data existed. When labels are sticky, a stale model falls out of
step with the *current regime* and appears to decay, while any detector firing
on distributional change tracks that regime. Under that account a detector
advantage would be momentum, not concept drift.

**Mitigation, added before the confirmatory run:** every comparison reports a
persistence reference — the AUC of predicting the previous label, which learns
nothing — and any dataset where that reference comes within 0.02 AUC of the best
policy is excluded. Both confirmatory datasets cleared it by a wide margin
(0.7235 vs 0.9690; 0.6198 vs 0.8834).

---

## 5. Exploratory result, and why it was not trusted

Robustness grids on `electricity` and `covertype` varied window count (30/45/60),
detector (KS/PSI) and memory depth (2/3/5), reporting mean C − B where retrain
counts matched exactly.

Both **failed** the stability rule fixed in advance: the sign of C − B flipped
across configurations. Stratifying afterwards showed the flips were not
scattered:

| | positive | mean C − B |
|---|---|---|
| electricity · PSI | 9/9 | +0.0298 |
| covertype · PSI | 9/9 | +0.0293 |
| electricity · KS | 7/9 | +0.0101 |
| covertype · KS | 6/9 | +0.0039 |

PSI was 18/18 positive on two independent datasets, with means agreeing to
0.0005. Every flip came from KS.

**That stratification is post-hoc** — chosen after seeing the results, on the
data that suggested it. It was recorded as a hypothesis, not a finding.

---

## 6. Confirmatory test

Pre-registered before any confirmatory data was generated, with `electricity`
and `covertype` barred **in code** from the confirmatory set.

> **H1.** With a PSI detector, mean C − B > 0.
> **H2.** With a KS detector, it is not reliably positive.
>
> **Decision rule.** *p* = fraction of PSI configurations with C − B > 0,
> pooled. p ≥ 0.90 and mean > 0.01 → supported. p ≤ 0.60 → refuted. Otherwise
> inconclusive, and it stays inconclusive.

| | PSI positive | mean C − B |
|---|---|---|
| Exploratory — electricity, covertype | **18/18** | +0.0296 |
| Confirmatory — gas-drift, bank-marketing | **4/12** | **−0.0023** |

**p = 0.333. H1 is refuted.** The exploratory pattern did not replicate.

KS: 3/9 positive, mean −0.0039 — not reliably positive, as H2 predicted.

*A drafting error worth recording: the pre-registration said "the same
thresholds decide H2" without specifying the direction mapping, so a mechanical
application labels H2 "refuted" when its prediction in fact held. The rule was
sloppily worded; the observation is unambiguous.*

---

## 7. Conclusion

> **A drift alarm does not carry timing information that a schedule of equal
> frequency lacks.** This holds across four synthetic regimes with known ground
> truth, and survives a pre-registered confirmatory test that a promising
> exploratory pattern failed.

Retraining itself is valuable where the boundary genuinely moves — up to
+0.196 AUC in these streams. All of that value comes from retraining. None of it
is attributable to the signal that triggered it.

**Practical consequence.** If you are running drift monitoring to decide *when*
to retrain, a fixed schedule at the same frequency will do as well. Monitoring
may be worth keeping for other reasons — auditability, diagnosing what changed,
regulatory evidence — but the retraining-trigger justification is not supported
here.

---

## 8. Limitations

**Two confirmatory datasets, and one contributed little.** `bank-marketing`
produced only 3 of 18 configurations with exactly-matched retrain counts — its
KS threshold saturated at 1.0 and still over-fired, meaning many windows have
nearly disjoint feature distributions. The confirmatory evidence rests mostly on
`gas-drift`.

**The benchmark pool is thin.** Only four of eleven streams could test the
question at all, and those four cluster in a narrow autocorrelation band. A
finding built on them inherits that narrowness.

**One detector family, one model.** KS and PSI on feature marginals, with
logistic regression as the learner. Multivariate detectors, or performance
estimation methods such as CBPE, are untested here.

**Not separated: timing versus stability.** The schedule's performance varies by
0.036 AUC across *phases* at a fixed retrain count, while the detector's does
not. Some of any detector advantage could be robustness to a tuning choice
rather than better timing. Distinguishing them needs a schedule tuned over phase
as well as period, and was out of scope.

**Synthetic regimes are mine.** The four drift regimes were designed by me, and
a reader may reasonably say the detector was set up to fail there. The real-data
work exists to answer that, and the confirmatory arm is what carries the weight.

---

## 9. Reproducing

```bash
pip install -r requirements.txt

# synthetic regimes with known ground truth
python3 analysis/run_comparison.py

# gate the catalogue: which streams can test anything?
python3 analysis/prepare_streams.py --all --windows 40

# confirmatory set only — excludes the hypothesis-generating datasets
python3 analysis/prepare_streams.py --all --confirmatory --windows 40

# robustness grid for any catalogued dataset
python3 analysis/robustness.py gas-drift

# NYC taxi, if you want the non-degradation result
python3 analysis/prepare_taxi.py --months 2024-01:2025-12 --tip-threshold 0.25
```

`PREREGISTRATION.md` records the hypothesis, the decision rule, the exclusions
and the confound, all committed and tagged before the confirmatory data existed.
