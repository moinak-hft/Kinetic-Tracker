# Motion association and scene-cue refinement

This is a training-free extension of the project's C++ tracker. It combines
established motion/assignment ideas and evaluates explicit hypotheses. It does
not claim a new state-of-the-art tracker or reproduce all of OC-SORT.
The frozen protocol and results determine the recommended operating modes.

## Matching still uses the custom Hungarian implementation

`kinetic::hungarian()` remains in `src/tracker.cpp`. Every association pass
constructs a gated bipartite graph, with explicit unmatched alternatives. The
assignment first maximizes finite feasible matches, then minimizes their cost.
Forbidden edges are removed **before** optimization. For T tracks and D
detections the augmented assignment is O((T+D)^3), often abbreviated cubic
Hungarian/Kuhn–Munkres matching. Matching is not a learned neural network.

The original `sort` and `two_stage` modes keep their existing numerical path.
`two_stage` associates high-score detections first, then unmatched established
tracks against low-score detections. Only unmatched high detections create
tracks. It is inspired by ByteTrack's confidence staging, not its full official
implementation. The actual official implementation is evaluated separately.

## Four-dimensional motion cost

The eight-state constant-velocity filter represents
`[cx,cy,w,h,vx,vy,vw,vh]`. Its transition depends on elapsed timestamps.
The observed box is `z=[cx,cy,w,h]`. Candidate innovation distance is
`d²=(z-Hx)'(HPH'+R)^-1(z-Hx)`. Cholesky solves avoid explicit inversion.
Class mismatch, nonfinite state and implausible centre displacement reject an
edge. The strict `motion` profile additionally requires `d² <= gate`.

Its cost is `(1-w)*(1-IoU) + w*d²/gate + direction_weight*(1-cos(theta))/2`.
The direction compares the last **two real observations** with the proposed
new displacement. Tiny or undefined displacements contribute zero. Predictions
are not treated as additional real observations. Below-IoU recovery candidates
also require a mutual row/column cost margin to avoid ambiguous reattachment.

The hybrid option allows above-threshold IoU pairs to bypass the guessed
Mahalanobis/cost-limit gate. It clips their normalized motion cost at one,
while preserving class, geometry and centre-distance bounds. Below-IoU pairs
retain strict uncertainty and ambiguity guards. This fallback was added during
validation after strict gating failed. All failed profiles remain in the report.

Gate values 13.2767 (four dimensions) and 9.2103 (two dimensions) are **heuristics**
here. Their resemblance to chi-square quantiles does not establish calibrated
coverage: process and measurement noise are not learned/calibrated on this data.

## Recovery after missing observations

After a multi-step gap, motion mode can blend velocity toward the displacement
between the last real box and the current real observation divided by elapsed
time. The blend lies in [.25,.8], and the velocity innovation is bounded.
Endpoint estimates share observations with the filter correction; they are
not independent measurements. Velocity variance therefore does not shrink from
this deterministic update. Joseph-form correction, symmetry and conservative
diagonal stabilization guard covariance validity.

The separate `recovery` mode is more conservative: it executes the original
two IoU confidence passes unchanged, then runs a third assignment **before
births**, only for unmatched confirmed tracks with two real observations and a
genuine missed/skipped interval, against unmatched high-score detections. It
uses a two-dimensional centre innovation gate plus size ratio, centre-distance,
direction/cost and mutual-margin guards. Only successful third-pass pairs get
the endpoint update. This prevents a weak recovery edge from displacing an
existing strong IoU match. It did not earn the default quality recommendation
on validation; keeping this result is part of the experiment.

This is not OC-SORT's complete observation-centric re-update (ORU). There is
no invented sequence of intermediate measurements, optical-flow network,
appearance embedding or access to future detections.

## Scene-change cue and scheduling

`SceneChangeDetector` consumes raw pixels before the current detector decision.
It keeps prior-only residual mean/variance/frequency statistics, compensates
global brightness offsets, and can align bounded translations using guarded
phase correlation. Translation needs distributed texture support and an actual
residual improvement. It is not general camera-motion compensation.

Confirmed predicted boxes are masked; tentative and lost boxes cannot hide a
new arrival indefinitely. Border regions created by alignment are excluded.
First-frame initialization is marked warming; dimension changes/strong cuts
reset statistics and request detection subject to the token budget. The analysis
resolution caps both axes. Logged raw novelty, robust novelty, global change,
reset and warmup signals make trigger behaviour inspectable.

Recurring background suppression can also suppress a weak target entering the
same moving region. Nonuniform illumination, rotation and perspective changes
remain failure cases. Fewer scene triggers are not proof of better tracking.
The robust cue remains optional because validation did not select it as the
best budget schedule.

`risk_scope=active` measures uncertainty/overlap on current confirmed eligible
tracks, avoiding stale lost-track saturation. The historical `all` scope is
preserved for reproducibility. Empty active uncertainty remains high to allow
rediscovery. Neither scope guarantees target detection.

The capacity-two scheduler token bucket enforces the recorded prefix budget.
Its exact-K uniform comparator is retrospective: it matches final call count,
not online knowledge or identical prefixes. Cached replay measures association
quality and call decisions; separate uncached runs measure live CPU runtime.

## Tests and attribution

Tests cover crossing/ambiguous motion, class/size gates, irregular timestamps,
low-score restrictions, preservation of IoU matches, recovery eligibility,
PSD/finite covariance, active-risk scope, recurring backgrounds, weak arrivals,
brightness, bounded translations, cuts and future-input invariance. A rebuilt
merged-v2 executable supplies byte-level regression controls.

References: [SORT](https://arxiv.org/abs/1602.00763),
[ByteTrack](https://arxiv.org/abs/2110.06864),
[Deep SORT](https://arxiv.org/abs/1703.07402),
[OC-SORT](https://arxiv.org/abs/2203.14360), and the official pinned sources in
`third_party/PROVENANCE.json`. These techniques are attributed prior work;
the contribution here is implementation, controlled comparison and failure analysis.
