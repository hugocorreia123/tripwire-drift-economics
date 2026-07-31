# Tripwire — Pre-registration for the confirmatory test

**Written before any confirmatory data was generated. Committed and tagged
before the run.**

---

## Where this came from

The exploratory phase ran a robustness grid on two datasets that passed the
usability gates (real degradation, tolerable autocorrelation): `electricity`
and `covertype`. Each grid varied window count (30/45/60), detector (KS/PSI)
and memory depth (2/3/5), reporting mean C − B at operating points where
retrain counts matched exactly.

Both datasets **failed** the pre-stated stability rule: the sign of C − B
flipped across configurations, so by that rule neither result is reportable.

Stratifying afterwards showed the flips were not scattered:

| | positive | mean C − B |
|---|---|---|
| electricity · PSI | 9/9 | +0.0298 |
| covertype · PSI | 9/9 | +0.0293 |
| electricity · KS | 7/9 | +0.0101 |
| covertype · KS | 6/9 | +0.0039 |

Every sign flip came from KS. PSI was 18/18 positive, and its means on two
independent datasets agree to 0.0005.

**That stratification is post-hoc.** It was chosen after seeing the results,
on the same data that suggested it. It is a hypothesis, not a finding, and
the numbers above cannot be used as evidence for it.

---

## Hypothesis

> **H1.** With a PSI drift detector, retraining triggered by the detector
> outperforms a schedule matched on retraining frequency: mean C − B > 0.
>
> **H2.** With a KS detector, it does not: mean C − B is not reliably
> positive.

Directional, because the exploratory phase gives a direction.

---

## Method, fixed in advance

- **Eligible datasets.** Any stream in the catalogue that passes both
  usability gates: retraining gain > 0.02 AUC, persistence ≤ 0.90.
- **Excluded from the confirmatory set.** `electricity` and `covertype`,
  which generated the hypothesis. They may be re-reported as exploratory,
  never as confirmation.
- **Grid.** Identical to the exploratory one: windows 30/45/60 × memory
  2/3/5, per detector. Operating points k ∈ {4, 6, 8, 10}, retaining only
  those where the detector's firing count matches the schedule's exactly.
- **Statistic.** Mean C − B per configuration; the summary is the fraction
  of configurations with C − B > 0, per detector.

## Decision rule, fixed in advance

Let *p* be the fraction of PSI configurations with C − B > 0, across all
confirmatory datasets pooled.

- **p ≥ 0.90 and mean C − B > 0.01** → H1 supported. Report that PSI-triggered
  retraining carries timing information a matched schedule does not.
- **p ≤ 0.60** → H1 refuted. Report that the exploratory pattern did not
  replicate, and that the ELEC2/covertype result was a false lead.
- **otherwise** → inconclusive. Report as inconclusive; do not reinterpret.

The same thresholds applied to KS decide H2.

**If fewer than two datasets pass the usability gates, the confirmatory test
does not run.** A result from a single stream would repeat the exact weakness
this exercise exists to fix, and "we could not find enough streams that
degrade" is itself a reportable outcome.

---

## What will be reported regardless of outcome

The number of catalogued datasets that passed the usability gates, and the
number that failed. Four of six streams tested so far — two NYC taxi windows,
`airlines`, `poker` — showed no meaningful degradation from a stale model.
That is a substantive observation about the availability of benchmarks
suitable for drift research, independent of anything this test concludes.

---

## What this cannot establish

Whether PSI's advantage, if it survives, reflects better drift timing or
merely greater stability with respect to a tuning choice the schedule is
sensitive to. The exploratory phase found the schedule's performance varies
by 0.036 AUC across phases at a fixed retrain count while the detector's does
not, which is an alternative explanation this design does not separate.
Distinguishing them would need a schedule tuned over phase as well as period,
and is out of scope here.
