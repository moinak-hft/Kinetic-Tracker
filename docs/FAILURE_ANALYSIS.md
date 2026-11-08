# What improved and what did not

The nine additional GMOT40 clips were declared before scoring. No threshold,
algorithm or recommended profile was changed after seeing these results. This
analysis explains measured differences; it does not establish their causal
mechanism from aggregate metrics alone. Scores below are TrackEval points on
0–100 scales, using identical author detections and every-second-frame detection.

| Case | v2 / hybrid HOTA | v2 / hybrid AssA | v2 / hybrid DetA | v2 / hybrid IDSW | v2 / hybrid FP | v2 / hybrid FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| balloon-2 | 51.71 /57.89 | 54.92 /62.48 | 49.17 /54.20 | 57 /3 | 339 /207 | 631 /479 |
| car-2 | 27.61 /25.91 | 52.78 /44.49 | 14.46 /15.10 | 2 /3 | 130 /123 | 2780 /2745 |
| fish-2 | 22.13 /20.67 | 33.02 /28.79 | 15.20 /15.25 | 17 /33 | 285 /266 | 5291 /5285 |

On balloon-2, association consistency and detection coverage both improve.
On car-2 and fish-2, detection quality slightly increases while association
quality falls. Fish identity switches nearly double. Thus lower false-negative
counts alone would give the wrong conclusion about overall tracking quality.
The nine full comparison videos retain the failure frames, predictions and
detector-call counters. Numeric IDs in separate methods are not correspondences.

Across nine clips, hybrid motion at fixed2 improves mean HOTA 35.82 → 36.55 at
893 calls, with six positive and three negative deltas. The mean gain is modest,
and this is one clip per category. It is not a full GMOT40 leaderboard result.
The unchanged two-stage profile remains the every-frame quality default because
none of the more complex validation candidates earned that recommendation.

## Scheduling is not consistently better than uniform spacing

The validation-selected adaptive policy uses860/1,781 calls on the additional
cohort, with mean HOTA 36.53. Its exact 860 call uniform comparator, with the same
hybrid association and other settings, scores 36.66. The adaptive mean delta is
therefore −0.14 points. This negative result remains in the primary tables.

The new robust scene cue and active-risk scope are controlled alternatives,
not automatic improvements. Novelty-trigger reduction can suppress nuisance
motion and also suppress weak arrivals. A token budget enforces compute usage;
it cannot make an inadequate detector or ambiguity signal reliable.

## What would be a defensible next research step

1. Calibrate motion/noise and identity ambiguity on a **new, declared validation
   cohort spanning several categories**, then use additional untouched recordings
   for confirmation. The current nine clips cannot become a fresh blind test again.
2. Inspect labelled mismatch intervals to distinguish detector misses,
   inaccurate boxes, crossing ambiguities and long gaps. Test individual changes
   against fixed-cadence and exact-count controls, retaining per-category deltas.
3. If testing appearance embeddings or optical flow, use a pinned pretrained
   model and report its training provenance, extra inference cost and failure
   cases. Calling an added model or feature research novelty is insufficient.
4. Acquire independently recorded target-domain footage with raw images,
   ground-truth identities and verified timing before ecological, insect
   generalisation or real-world latency claims.

No such future work is claimed as implemented. The completed contribution is a
tested C++ system, attributable methods, preserved controls and a reproducible
study that can expose both gains and failures without new detector training.
