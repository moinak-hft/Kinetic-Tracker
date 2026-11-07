# Related work and the question this branch tests

This branch investigates **how to spend detector calls while preserving object
identity and discovering new, short-lived targets**. The intended application is
small-object/insect tracking with a fixed camera and limited compute. Its current
contribution is an inspectable C++ implementation and reproducible experiments.
It does not establish a new state-of-the-art tracker or a new scheduling principle.

## Existing tracking methods

**SORT** combines Kalman motion estimation and frame-to-frame assignment as a
simple online baseline. Its authors explicitly describe limitations around
occlusion and re-entering objects. The project's corrected Kalman/assignment
baseline is SORT-style; extra gating and lifecycle choices are specified in the
implementation rather than being represented as a reproduction of every original
SORT detail. See [the paper](https://arxiv.org/abs/1602.00763) and
[the authors' implementation](https://github.com/abewley/sort).

**ByteTrack** associates low-confidence detections to existing tracks after the
high-confidence association stage. This can recover partially visible objects
that would otherwise be discarded. The core's confidence-based second stage is
inspired by this established idea. It is not itself labeled official ByteTrack.
A separate pinned reference runs the authors' tracking code with the compatibility
edits recorded in `third_party/PROVENANCE.json`. See
[the paper](https://arxiv.org/abs/2110.06864) and
[the official repository](https://github.com/FoundationVision/ByteTrack).

**OC-SORT** studies motion-estimation errors accumulated during occlusion and
uses observation-centric updates to improve association under nonlinear motion.
It is an important motion-only comparator for tiny objects whose appearance
features may be weak. The pinned tracking-only reference keeps the original
association/Kalman logic. See [the paper](https://arxiv.org/abs/2203.14360) and
[the official repository](https://github.com/noahcao/OC_SORT).

**BoT-SORT** combines motion and appearance information with camera-motion
compensation. It is relevant for moving-camera extensions. It is not a required
dependency in this fixed-camera experiment, and a pedestrian appearance model is
not assumed to identify insect instances. See
[the paper](https://arxiv.org/abs/2206.14651) and
[the authors' repository](https://github.com/NirAharon/BoT-SORT).

## Scheduling is established prior work

**DriftSched: Covariance-Guided Adaptive Detection Scheduling for Efficient
Multi-Object Tracking**, MECO 2026, already uses Kalman covariance and spatial
overlap risk to allocate detection under a compute budget. Its abstract explicitly
describes rate-matched evaluation against skipping/heuristic baselines. Therefore
"run detection when Kalman uncertainty grows" is not a novelty claim for this
project. See [the primary IEEE publication](https://ieeexplore.ieee.org/abstract/document/11579050),
DOI [10.1109/MECO70748.2026.11579050](https://doi.org/10.1109/MECO70748.2026.11579050).

The working hypothesis here is more specific: **a scheduler that only monitors
existing tracks can miss newly arriving insects, even while its current tracks
remain confident**. Cheap whole-frame image-change evidence and bounded full-frame
refreshes may improve discovery relative to existing-track-only scheduling at the
same realized detector-call rate. This is a testable extension hypothesis, not a
verified claim that nobody has studied new-arrival scheduling before.

The novelty signal must come from current raw pixels or another explicitly
available cheap sensor. Looking at the cached detector boxes of a skipped frame,
ground truth, or future frames would leak the answer. The current implementation
uses foreground/image-change evidence and logs decisions so that this condition
can be checked. It makes frame-level detect/skip decisions: ranking individual
tracks does not reduce a shared whole-frame detector's cost by itself.

## Evaluation and dataset scope

[HOTA](https://arxiv.org/abs/2009.07736) evaluates detection and identity association
across localization thresholds. [TrackEval](https://github.com/JonathonLuiten/TrackEval)
provides the official metric implementations used here. HOTA, AssA, DetA and IDF1
are reported together with ID switches, false positives and false negatives.
The documented upstream CLEAR fragmentation limitation is retained transparently.
Means across sequences are explicitly unweighted sequence means, not official
pooled benchmark scores. Tracker-only cached replay timings are separated from
real detector pipeline timings.

The [Australian Native Bee Detection and Tracking Dataset](https://researchdata.edu.au/australian-native-bee-tracking-dataset/4055838)
([DOI 10.26180/32315067.v1](https://doi.org/10.26180/32315067.v1)) supplies insect
boxes and identities. The available images contain processed ROI backgrounds and
annotation overlays, and some source frame indices are absent. Source-derived
timestamps are preserved; the official tracking references retain fixed update
steps, while the C++ engine can consume physical time deltas. These data support
association and processed-image probes, not a claim of unprocessed live-camera
deployment. Ground-truth-derived detector inputs are always labeled oracle
association diagnostics. Splits separate entire recording days, but share a camera
and hive, so they do not establish cross-camera generalization.

[GMOT-40](https://spritea.github.io/GMOT40/)
([CVPR paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Bai_GMOT-40_A_Benchmark_for_Generic_Multiple_Object_Tracking_CVPR_2021_paper.pdf))
includes varied object categories and insect sequences. Its original task includes
one-shot generic tracking; an experiment using supplied author detector caches is
reported as an association/scheduling study with those inputs, not as an equivalent
reproduction of the full generic detector benchmark. The sequence subset and split
are declared in the data manifests and `configs/splits.json`.

The project's synthetic videos isolate crossings, turns, occlusion, births and
background motion. Their detector boxes are simulator-generated perturbations,
not neural inference. Validation/test seeds are independent within the same
scenario families; this does not demonstrate unseen-family generalization.

Evidence for the scheduler requires matched realized detection rates, results on
held-out sequences, a comparison against fixed/uncertainty-only schedules, discovery
delay and missed short-lived targets, and visible failures. A detector upgrade,
extra UI or a favorable single clip is insufficient to establish that hypothesis.
