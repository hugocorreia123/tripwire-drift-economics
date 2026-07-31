# Tripwire

**Your drift monitor fires. Is that worth more than a calendar reminder?**

Measured across four synthetic regimes with known ground truth and eleven real
data streams: **no.** A schedule that retrains at the same frequency does just as
well, and sometimes better.

---

## The short version

Production ML teams watch feature distributions and retrain when something
shifts. The pitch is that monitoring tells you *when* — that it catches the
moment your model went stale, so you retrain then rather than on some arbitrary
calendar.

That pitch is almost never tested, because the usual comparison is *monitored*
against *unmonitored*. And retraining helps for two quite different reasons:

- fresher data helps, **whenever** you do it
- retraining **at the right moment** helps

Only the second belongs to the detector. Comparing monitored against unmonitored
credits the alarm with both.

So this study matches them on cost. Three policies, **identical retraining
frequency**, differing only in *when* they fire: never, a fixed schedule, and a
detector tuned to fire at the schedule's rate. Whatever the detector wins is
timing information, because everything else is held equal.

---

## What was found

**1. On synthetic streams where the truth is known, the signal adds nothing.**
The schedule wins or ties in every regime. Where concept drift is invisible to a
feature detector, the detector is significantly *worse* — it fires on noise, at
random times, and random timing beats nothing while losing to a regular one.

**2. Most drift benchmarks cannot test this.**
Of eleven real streams, **only four degraded a stale model at all**. The rest
tie for an uninteresting reason: nothing decayed, so nothing could help. Across
the COVID collapse, NYC taxi volume fell **97%** — and tip prediction barely
moved. Enormous distribution shift, no concept drift. Every monitor in
production would have fired; nothing needed doing.

**3. A promising pattern appeared, was pre-registered, and died.**
On two datasets a PSI-based detector looked like it helped: 18/18 configurations
positive, means agreeing to 0.0005. That was noticed *after* the fact, so it was
written up as a hypothesis with a decision rule fixed in advance and tested on
two datasets held out from its discovery. **It did not replicate** — 4/12
positive, mean −0.0023, refuted under the pre-registered rule.

**4. The streams that degrade are the autocorrelated ones.**
Degradation and label stickiness correlate at **r = +0.78**. Every stream that
passed the usability gate sits at persistence 0.80–0.87; every one excluded for
not degrading sits below 0.67. That confound was found before the confirmatory
run and a persistence exclusion was added in response.

---

## What to do about it

- **If monitoring exists to schedule retraining, a calendar does the same job.** At matched frequency the alarm added nothing here.
- **Monitoring may still earn its keep** for auditability, for diagnosing what changed, or for regulatory evidence. Those uses are untouched by this result.
- **Check your stream degrades before drawing conclusions from it.** Four of eleven standard benchmarks cannot support a drift-response experiment at all.
- **Check the persistence baseline.** If predicting the previous label scores near your model, you are measuring momentum, not concept drift.

---

## The design

```
A  never       train once, never update
B  scheduled   retrain every k windows
C  triggered   retrain when the detector fires, threshold calibrated
               so it fires at the same average rate as B
```

Identical compute, identical retrain rule, identical memory window. Only the
timing differs. Evaluation is prequential — each window is scored *before* any
retraining it would trigger — and scored by **AUC** rather than accuracy,
because base rates here run as high as 0.88 and accuracy cannot tell "no signal"
from "signal present and stable".

---

## Reproducing

```bash
pip install -r requirements.txt

python3 analysis/run_comparison.py                    # synthetic regimes
python3 analysis/prepare_streams.py --all             # gate the catalogue
python3 analysis/robustness.py gas-drift              # sensitivity grid
```

`PREREGISTRATION.md` holds the hypothesis, the decision rule and the exclusions,
committed and tagged before any confirmatory data existed.

---

## Layout

```
FINDINGS.md                      the full report
PREREGISTRATION.md               hypothesis and decision rule, fixed in advance
analysis/drift_stream.py         synthetic streams with controllable drift
analysis/policies.py             the three policies, detectors, calibration
analysis/run_comparison.py       synthetic and real comparison
analysis/prepare_streams.py      OpenML catalogue with usability gates
analysis/prepare_taxi.py         NYC TLC loader
analysis/robustness.py           sensitivity grid over analysis choices
analysis/frontier.py             best schedule vs best detector at equal cost
```

---

## Part of a series

Three studies applying the same discipline to questions practitioners answer by
folklore:

- [**Prism**](https://github.com/hugocorreia123/prism-token-taxes) — do LLM token-saving techniques actually save money?
- [**Tolerance**](https://github.com/hugocorreia123/tolerance-agent-evals) — could your evaluation detect the improvement you claim? (It audits Prism, and finds Prism's headline sits below its own detection threshold.)
- **Tripwire** — is a drift alarm worth more than a cron job?

Each reports what it found, including when that was nothing.
