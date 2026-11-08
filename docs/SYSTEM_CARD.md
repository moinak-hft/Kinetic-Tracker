# Kinetic-Tracker system card

## Intended use

Kinetic-Tracker is an educational native C++ multi-object tracking engine and a
reproducible experiment platform. It studies association, missed observations,
detector scheduling and output policy on recorded footage and explicit synthetic
diagnostics. Its evidence supports the particular measured configurations and
clips, not guaranteed deployment reliability or a universal tracking improvement.

The current operating recommendations are in `configs/recommended_profile.json`.
The quality association is selected with every-frame detection. The budget
association is selected separately under fixed-2 detector availability using the
same declared source-balanced criterion and raw-insect non-regression guard.
The budget schedule is then selected from the declared validation options. Those
choices are frozen before additional confirmation scoring; this card deliberately
does not hardcode a winner that could disagree with the recorded artifacts.

This is not an autonomous pest-control decision system, a verified ecological
counting instrument, a hardware implementation, or a claim of a new tracking
algorithm. A tracking ID is a temporal association hypothesis, not a verified
biological identity.

## Detector, learning and data provenance

The original single-class `pest` ONNX detector is retained without retraining.
Its SHA-256 is:

```text
2c938174e9ff98a94d2305c8918f2511febe1bd214a14819092f76cb26e7e47b
```

The uncached engine uses OpenCV DNN with the OpenCV CPU backend. It performs
documented letterbox preprocessing and inverse coordinate conversion. The model's
original training images, splits and complete training procedure are unavailable,
so its scores cannot establish clean held-out detector generalisation.

The refinement quality study does **not** pretend that all evaluation detections
come from that YOLO model:

- GMOT40 natural-image experiments use the authors' fixed, released one-shot
  detector outputs. The first-frame exemplar is part of that source protocol.
- Synthetic experiments use rendered scenes and simulated noisy detections.
- Monash native-bee oracle diagnostics use identity-free ground-truth boxes.
  The source JPEGs have ROI masks and counters, which confound natural
  scene-change discovery. Those results remain diagnostic only.

There is no new neural-network training in this refinement. Validation chooses
heuristic thresholds, association profiles and scheduling configurations. This is
parameter calibration/model selection, not learned feature training. Online
recurring-motion statistics adapt causally from earlier frame differences; they
are neither supervised labels nor a trained insect-recognition model.

See [DATA.md](DATA.md), the data provenance JSON files and the report's source
manifest for dataset licences, hashes, conversions, exclusions and nominal timing.
GMOT40 imagery is CC BY-NC-SA 4.0; the Monash release is CC BY 4.0. The code licence
does not relicense source images or their derived comparison videos.

## Tracking state and emission policy

The motion state is an eight-dimensional constant-velocity model for box centre,
width, height and their four velocities. Prediction uses monotonically increasing
timestamps and reconstructs the complete box. Invalid geometry is rejected.
Association uses class compatibility, pre-assignment gates and finite
minimum-cost matching with explicit unmatched tracks/detections.

Tracks are tentative, confirmed or lost:

- High-confidence unmatched detections can initialize tracks. Confirmation depends
  on the configured hit requirement; the frozen study uses its recorded value,
  which can differ from the API's default.
- A deliberate detector skip propagates state and does not by itself mark a
  confirmed track lost. There has been no detector evidence of absence.
- A detector pass without an accepted association can mark an established track
  lost. Reassociation can reactivate it before expiry.
- Deletion and prediction-output horizons use elapsed time since a real
  observation. Internal retention is separate from published output.

Telemetry records tentative states, observation/prediction flags and the last
detector score. The score is not a calibrated track probability.

`output_policy=active`, used in primary controlled comparisons, emits confirmed,
non-lost tracks inside the output horizon. It can emit predictions during deliberate
skips. `output_policy=all` additionally allows retained lost predictions inside
that horizon; tentative tracks are still excluded from MOT metrics. All-output
diagnostics are reported separately. Changing emission policy changes false
positives/false negatives, so it must not be mistaken for an association gain.

## Association options and uncertainty limitations

`sort` is the high-confidence IoU baseline. `two_stage` adds lower-confidence
association for existing confirmed tracks. This is ByteTrack-inspired behavior,
not the complete official ByteTrack implementation.

The optional `motion` family adds covariance-scaled distance, observed direction
and guarded post-gap endpoint-velocity correction. Hybrid candidates preserve
accepted-overlap possibilities while keeping bounded non-overlap recovery.
The separate `recovery` candidate preserves the ordinary two-stage passes and
attempts a guarded third association pass only for remaining confirmed gap tracks
with sufficient real observation history. These are declared experimental
candidates, not full OC-SORT observation-centric re-update.

Process noise **Q** and measurement noise **R** use hand-specified box-scale rules.
They were not fitted to a validated sensor-error distribution. Mahalanobis gates
with chi-square-shaped numerical values therefore do **not** establish calibrated
99% coverage, confidence, or identity guarantees. The motion backend stabilizes
covariance numerically and uses a conservative correction; that does not repair
an incorrect motion model during abrupt turns, occlusion or poor detections.

Strict motion-gating failures, later validation amendments and unsuccessful
recovery candidates remain in the validation tables. A candidate can rank
differently when the detector runs less often. Recommendations follow the frozen
quality and budget selection rules rather than the most attractive isolated clip.

## Detector scheduling and causal scene signals

Fixed, uniform, random and risk-based schedules are available. Budgeted schedules
obey a hard prefix detector-call cap. The cap takes priority over risk, maximum-gap
or scene-reset requests; it does not borrow future computation. An exact-count
uniform control is given the adaptive method's final call count. It is a useful
retrospective, content-independent comparator, not an online method that predicts
the future call count.

Risk can use all retained internal tracks or only confirmed, non-lost estimates
inside the output horizon. Active-only risk limits stale-ghost influence. When no
eligible active track exists, the uncertainty cue requests discovery; the compute
budget can still delay an actual detector pass.

The robust scene heuristic reads current raw pixels and past tracker state before
current detections are available. It uses:

1. Grayscale analysis with both image dimensions bounded by the analysis edge.
2. Robust global-brightness compensation.
3. Guarded small global-translation compensation, accepted only with distributed
   texture support and improved residual alignment; warped borders are excluded.
4. Per-pixel residual/frequency histories scored **before** incorporating the
   current residual, reducing repeated background-motion triggers.
5. Masking of current confirmed regions, not tentative or lost ghosts, followed by
   a minimum-component-area filter.

Initial warmup is explicit. Strong scene discontinuities and dimension changes
reset histories and request refresh. The default translation bound is six pixels
at analysis resolution. The method does not compensate arbitrary rotations,
perspective changes, large camera movement, rolling shutter or parallax.

Important failure tradeoffs remain: a modest-contrast arrival inside a repeatedly
moving region can be suppressed; a confirmed but incorrect predicted region can
hide evidence; small targets can disappear through downsampling/component
filtering; moving shadows and nonuniform illumination can trigger requests.
Reduced novelty activation is a nuisance diagnostic, not proof of better identities
or complete new-object discovery.

## Evaluation scope and timing

The refinement uses explicit stages:

- **Validation:** source-balanced comparison of six synthetic scenarios and a
  natural insect clip. Quality and budget selection are separate, declared stages.
- **Reused exploratory evidence:** earlier insect and synthetic test clips had
  already been inspected. Their reuse is disclosed rather than relabelled as a
  clean untouched test.
- **Additional confirmation:** nine predeclared non-insect category clips,
  1,781 original frames, scored after the recommendation is frozen. These cover
  aircraft, balls, balloons, birds, boats, cars, fish, people and livestock, not evidence of
  insect-specific, new-site, recording-independent or whole-benchmark generalisation.
- **Diagnostics:** processed bee imagery and oracle boxes, clearly separated from
  detector and natural-scene discovery claims.

TrackEval metrics measure HOTA, association/detection quality, identity consistency
and error counts. Reported clip means are unweighted within the stated cohorts;
counts are summed and detector-call rates are frame-weighted. These are not
official pooled leaderboard scores. A small number of clips and correlated video
frames do not justify independent-frame significance claims.

Discovery diagnostics measure the first selected detector box matching an annotated
target at score/IoU thresholds. They do not guarantee correct track initialization
or identity. First annotated visibility is not verified physical object birth.

Some real datasets lack verified capture timestamps/FPS, or have gaps and upstream
processing. Their timestamp maps are explicitly nominal. Discovery delays in
**frames** are primary; physical velocity and real-camera deadlines cannot be
inferred from nominal playback rates.

Cached replay excludes detector inference. It can establish association/scheduling
quality and detector-call counts, not end-to-end neural-inference speedup or energy
savings. Separate uncached runtime records, when present, report their actual
timing scope and repeats. Initialization, warmup, output work, video close and
per-frame versus wall timing must remain distinguishable. A video's playback FPS
is not a throughput measurement.

## Reproduction and local execution

The experiment runner records source hashes, executable/evaluator hashes, protocol,
selection and confirmation freeze. The offline refinement report derives its
numbers from those saved artifacts; it never reruns a detector or selects a new
winner. Input/output collision guards protect declared inputs, and generated data,
build products and run outputs have separate directories.

On Windows, a downloaded ZIP can cause PowerShell to block `scripts/build.ps1`.
The supported alternative is the repository's Python build helper:

```powershell
.\.venv\Scripts\python.exe tools/build.py
```

It uses the existing CMake/MSYS2/OpenCV installation and runs the declared tests.
No PowerShell profile edit, execution-policy change, antivirus exclusion or
administrator elevation is needed. Follow the root reproduction guide for the
isolated Python environment and pinned dependencies.

Dataset preparation restricts sources to the documented author/university hosts,
checks pinned sizes/hashes and archive paths, and reads data rather than executing
downloaded dataset scripts. This reduces avoidable risk; it is not a guarantee that
any software or media parser can never contain a vulnerability.
