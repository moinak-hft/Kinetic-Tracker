# Method and experimental protocol

## Question and supported scope

The study asks whether current image-change evidence can help allocate a limited
number of whole-frame detector calls while preserving identities and discovering
short-lived targets. The engine is an online, single-camera C++ tracking-by-detection
implementation. The two-stage association is inspired by ByteTrack; it is not the
official ByteTrack algorithm. Official ByteTrack with documented compatibility
adapters and official OC-SORT are separate reference implementations.

Covariance-guided scheduling already exists, including DriftSched (MECO 2026).
This branch tests a small-object/new-arrival hypothesis; it does not establish
novel scheduling theory or a state-of-the-art result. See `RELATED_WORK.md`.

## Detector and tracker

The detector uses OpenCV DNN's CPU backend and an objectness-free YOLOv8 ONNX
export. Each source image is resized without changing aspect ratio and padded
with value 114 to a square, default 640 pixels. BGR input becomes normalized RGB.
The decoder supports the usual `[1,C,N]` / `[1,N,C]` layouts, selects the maximum
class score, reverses padding/scale, clips boxes to the source image and applies
class-wise NMS. It does not multiply by a separate objectness score. These are
float detector scores, not calibrated probabilities. No detector training occurs
inside the tracking engine.

Each track has an eight-dimensional state
`[cx,cy,w,h,vx,vy,vw,vh]`. Prediction uses the elapsed source timestamp and a
constant-velocity transition. Each coordinate/velocity pair receives a
constant-acceleration process-noise block proportional to
`[[dt^4/4,dt^3/2],[dt^3/2,dt^2]]`; noise scales are explicit geometry-dependent
heuristics, not learned uncertainty calibration. Measurements are `[cx,cy,w,h]`.

Association first forbids class mismatches and IoU values below the configured
gate. The custom rectangular Hungarian solver finds maximum-cardinality finite
matching, then minimum cost `1-IoU`; dummy assignments allow unmatched tracks and
detections. Applying the gate before optimization avoids rejecting an invalid
assignment after it has displaced a valid alternative. `sort` uses high-confidence
detections only. `two_stage` additionally matches remaining confirmed tracks to
low-confidence detections; only unmatched high-confidence detections spawn tracks.
There is no appearance embedding, optical-flow tracker, learned association or
camera-motion compensation.

Track state distinguishes tentative, confirmed and detector-confirmed lost states.
A deliberate detector skip supplies no negative observation and does not itself
mark a confirmed track lost. Real unmatched detector updates reset the hit streak
and mark confirmed tracks lost. Identity can be recovered until elapsed time since
the last actual observation exceeds `max_age_seconds`. Returned predictions have
a shorter output horizon; an internally retained track may therefore be absent
from returned telemetry while remaining eligible for matching and risk signals.

## Causal scheduling signals

At frame `n`, the engine first decodes the current image and predicts existing
tracks to its timestamp. Before accessing that frame's detector output it computes:

- **Uncertainty:** maximum over internally retained tracks of
  `sqrt(max(0,Pcx,cx + Pcy,cy))/max(1,hypot(w,h))`, clipped to `[0,1]`.
  An empty tracker returns 1. This is a relative motion-uncertainty heuristic.
- **Ambiguity:** maximum pairwise predicted IoU among retained tracks of the same
  class, or zero. This is an overlap proxy, not an association-error probability.
- **Image novelty, adaptive mode only:** BGR-to-gray, 3x3 Gaussian blur, absolute
  difference from the preceding available frame, then threshold at 22. Pixels
  inside non-tentative returned track boxes are masked with padding
  `max(3,ceil(.25*max(w,h)))`. The signal is the remaining changed-pixel count
  divided by the entire source image area. Lost boxes still eligible for returned
  tracks also mask this signal. The first frame's novelty is zero.

All these inputs are current/past estimates or pixels. Cached detections are
loaded as input data but the scheduler does not inspect the current frame's boxes
unless it elects to run detection. Ground truth, future pixels and future detector
outputs are not scheduler features. The image-change proxy can react to lighting,
vegetation, camera motion or processed ROI boundaries, and can hide new targets
near a masked existing box. A maximum-gap refresh is requested to limit reliance
on that proxy; this does not guarantee target discovery.

`covariance` requests a call for uncertainty, `risk` also uses ambiguity, and
`adaptive` additionally uses image novelty. Minimum-gap and maximum-gap requests
are evaluated in observation-index frames. `age` is currently an alias of `fixed`:
both use the gap since the last detector call, not an individual track's age.
`random` requests a Bernoulli call with probability `1/interval`; unlike the three
risk-based modes it has no hard rate cap. `every` and the offline matched `uniform`
schedule are additional explicit controls.

## Hard detector-call budget and matched control

The three risk-based schedules start with one token, refill by target rate `r`
per subsequent source observation and cap token storage at **two**. A call costs
one token. They also enforce, at every observation prefix `n`,

```text
C(n) <= floor(1 + r*(n-1))
```

Risk and maximum-gap requests never borrow tokens or override this cap. The code
requires `max_gap >= ceil(1/r)` to avoid an obviously infeasible refresh interval.
The initial call explains why a short sequence's realized `C(N)/N` may slightly
exceed `r`; `r` is not a demand to spend all available tokens. Calls are whole-frame
invocations, not independent per-track computations.

For each adaptive run with length `N` and realized count `K`, `--matched` runs a
content-independent comparator using exactly `K` frames:

```text
K=1: frame 1
K>1: frame(j) = 1 + floor(j*(N-1)/(K-1)), j=0,...,K-1
```

This is a **retrospective matched-total-budget control**: it knows the sequence
horizon and the adaptive run's final call count. It does not choose frames using
ground-truth quality, but it is not an online scheduling policy, and equality of
total calls is not equality of every prefix's resource allocation. Uniform versus
adaptive differences therefore isolate a useful placement comparison with that
qualification, rather than proving superiority to every feasible online baseline.

## Frozen primary study

`configs/splits.json` defines entire-sequence groups. Six synthetic scenario
families use independent validation/test seeds, not unseen test families. The
native-bee split separates recording days but shares camera/hive; supplied images
have processed ROI backgrounds, counters and sampling gaps. Their GT-derived boxes
are explicitly oracle association diagnostics. GMOT-40 uses the insect-only custom
split insect-0 train, insect-1 validation, insect-2/3 test and supplied author
one-shot detections. Its FPS is nominal/assumed; frame delays are more defensible
than verified physical milliseconds. These are not complete official leaderboard
protocols or field-deployment datasets.

Scheduler selection searches risk thresholds `.12,.18,.28` and novelty thresholds
`.00008,.0003`, separately at target rates `.25,.5`, over **synthetic validation
only**. It maximizes sequence-mean HOTA, breaking ties by IDF1 then fewer calls.
`configs/scheduler_selection.json` preserves every candidate. Both selected
policies use risk `.28`, novelty `.00008` and max-gap 8.

Frozen `configs/study.json` SHA-256:
`34b6972e02abece40eceee6ef774964ed829341bdc56994c6f3b01a4b757477b`.
Primary benchmark commands explicitly pass **`--output-policy all`**, min-hits 1,
IoU .1, prediction horizon .24 seconds, max age .75 seconds and high/low confidence
.35/.08. They do not inherit the current CLI's `active` output default.

## Output-policy diagnostic is post hoc

After initial held-out error inspection exposed many stale lost-track outputs,
an `active`/`all` diagnostic was added. `all` emits otherwise eligible confirmed
and lost predictions. `active` suppresses detector-confirmed `lost` boxes from
MOT results and rendering while retaining their otherwise eligible telemetry.
Predictions from confirmed tracks on intentional skipped frames still appear.
Tentative tracks are never emitted to MOT/rendering under either policy.

Validation then supported the new CLI default: native-bee oracle HOTA increased
from .6678 (`all`) to .8293 (`active`), and GMOT validation from .2408 to .2522.
This engineering choice was made **after held-out errors had already been seen**.
Its additional test results are exploratory diagnostics, not a pristine blind test
of that output-policy choice. Primary frozen-study results remain `all`, and
`configs/output_policy_diagnostic.json` / separate `runs/output_policy_*` folders
preserve the distinction. Suppressing stale output improves precision but can
remove useful occlusion predictions; it is not a change to motion estimation.

## Metrics and runtime are separate evidence

Pinned official TrackEval modules compute HOTA, AssA, DetA, IDF1 and CLEAR errors
on validated class-1 MOT inputs. Every supplied prediction is scored without a
confidence cutoff. The custom adapter removes unmarked GT but does not apply the
MOT pedestrian distractor/ignore-region protocol. Sequence scores are unweighted
means; counts are summed. `Frag` retains the documented upstream empty-frame caveat.

Discovery measures the first selected detector box with score at least .08 matched
one-to-one to a GT object at IoU at least .5. It measures **detector-box availability**,
not creation of a new high-confidence track. Delays start at first annotated
visibility, not a verified physical birth; short targets use first-to-last annotated
span at most .4 nominal seconds. Missed targets and frame delays are retained.

Quality results use cached inputs, so they cannot establish detector runtime.
`tools/runtime.py` separately executes uncached CPU inference on the original
unlabelled sample using three alternating-order repetitions per policy and a
declared thread count. Initialization/model loading and one warmup inference are
reported separately; preview/video encoding are disabled. Wall time includes
frame processing, logging flush and video finalization. Per-frame totals are
sampled before appending that frame's CSV log. Runtime does not establish quality
on this unlabelled sample, energy use, GPU performance, camera-to-display delay,
or a hard real-time deadline. See the measured runtime artifact for actual values.

The primary held-out `all` study remains mixed: synthetic HOTA is .7196 for
adaptive50 versus .7063 at the exact matched call count, but GMOT is .2452 for both;
adaptive25 is below its matched control on GMOT (.1830 versus .1892). Results on
native-bee oracle inputs do not establish raw-camera detector performance. See
`REFERENCE_RESULTS.md`, experiment manifests and the report for full comparisons.
