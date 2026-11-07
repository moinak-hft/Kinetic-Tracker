# Frozen reference comparisons

These are measured outputs from the first frozen held-out study. No reference
threshold was changed after viewing test results. The C++ study configuration
`configs/study.json` had SHA-256
`34b6972e02abece40eceee6ef774964ed829341bdc56994c6f3b01a4b757477b`.

HOTA/IDF1 below are **fractions and unweighted means across sequences**, not pooled
official benchmark scores. IDSW, FP and FN are summed counts. All tabled trackers
consume the available detector input at every observation index. This does not
execute the detector during cached replay or establish live FPS.

Reference settings are fixed in `tools/reference_benchmark.py`: high confidence
0.35, OC-SORT min_hits=1 / IoU=0.1 / direction inertia=0.2 / lookback=3, and original
ByteTrack matching threshold 0.8. Track age is nominally 0.75 seconds converted to
each upstream frame-count API. ByteTrack retains original low threshold 0.1 and
birth threshold high+0.1. The core has its own documented output/lifecycle rules;
the references are not artificially modified to match those rules.

## Held-out results

| Dataset/input | Tracker | HOTA | IDF1 | IDSW | FP | FN |
|---|---|---:|---:|---:|---:|---:|
| Synthetic, 6 sequences / 1,080 frames | C++ every_two_stage | 0.8016 | 0.8438 | 15 | 311 | 102 |
| Synthetic | Official OC-SORT | 0.6801 | 0.7869 | 8 | 0 | 522 |
| Synthetic | Official ByteTrack + compatibility adapters | 0.7556 | 0.8274 | 14 | 0 | 288 |
| GMOT-40 supplied one-shot detector caches, 2 insect sequences / 348 frames | C++ every_two_stage | 0.2930 | 0.4136 | 175 | 1,846 | 1,924 |
| GMOT-40 supplied detections | Official OC-SORT | 0.2707 | 0.3924 | 119 | 187 | 2,380 |
| GMOT-40 supplied detections | Official ByteTrack + compatibility adapters | 0.2343 | 0.3272 | 77 | 237 | 2,716 |
| Native-bee **oracle boxes**, 3 sequences / 1,517 available images | C++ every_two_stage | 0.5748 | 0.6922 | 83 | 1,236 | 6 |
| Native-bee **oracle boxes** | Official OC-SORT | 0.9394 | 0.9384 | 15 | 0 | 108 |
| Native-bee **oracle boxes** | Official ByteTrack + compatibility adapters | 0.7924 | 0.9120 | 13 | 1 | 172 |

The synthetic and small GMOT subset show a recall/false-positive tradeoff, not a
uniform accuracy win: the C++ engine has higher HOTA here but more false positives
and identity switches than some references. Native-bee oracle results expose a
substantial weakness in the current core's lifecycle/output choices. Telemetry
shows `lost`, `observed=0` predictions retained for up to the configured prediction
horizon, while the reference outputs are active-track-only. For example, the
first native-bee test clip has three GT objects versus four C++ outputs at frame 2,
then two GT objects versus five outputs at frame 3. Several stale predictions
remain after disappearance or a failed fast-motion match. These are scored false
positives, not evidence of an evaluator failure. Physical timestamp handling also
differs because the upstream references retain their original fixed Kalman steps.

The current adaptive50 policy equals fixed2 exactly on this GMOT held-out subset
(HOTA 0.2452, IDF1 0.3268, detector-call fraction 0.5029). Its image-change/risk
triggers under the budget therefore do not establish an adaptive advantage on
these clips. Both are below the full-detection core. The result is retained rather
than selecting a new threshold on the test data.

## Validation detector probes

| Validation input | Tracker | HOTA | IDF1 | IDSW | FN |
|---|---|---:|---:|---:|---:|
| GMOT-40 supplied author detections, insect-1 / 201 frames | OC-SORT | 0.2462 | 0.3636 | 92 | 1,482 |
| GMOT-40 supplied author detections | ByteTrack + adapters | 0.1751 | 0.2284 | 137 | 1,894 |
| Original ONNX detector on GMOT insect-1 | OC-SORT | 0.0822 | 0.1276 | 69 | 2,392 |
| Original ONNX detector on GMOT insect-1 | ByteTrack + adapters | 0.0260 | 0.0369 | 15 | 2,665 |
| Original ONNX detector on native-bee, 4 sequences / 497 images | OC-SORT | 0.0861 | 0.1246 | 2 | 743 |
| Original ONNX detector on native-bee | ByteTrack + adapters | 0.1059 | 0.1384 | 1 | 725 |

The weak original-ONNX results persist with two independent official trackers.
They point to detector inputs and domain/confidence calibration as practical
bottlenecks; these measurements alone do not distinguish training-data mismatch
from other detector issues. Changing only association cannot recover objects
absent from or rejected in the detector input. Native-bee's oracle-box experiment
is kept separate from these predictions.

## Reproduce and inspect

```bash
python tools/reference_benchmark.py --dataset synthetic --split test --source provided
python tools/reference_benchmark.py --dataset gmot40 --split test --source provided
python tools/reference_benchmark.py --dataset native_bee --split test --source provided
python tools/reference_benchmark.py --dataset gmot40 --split validation --source provided
python tools/reference_benchmark.py --dataset native_bee --split validation --source onnx
python tools/reference_benchmark.py --dataset gmot40 --split validation --source onnx
```

Each corresponding `runs/reference_<dataset>_<split>_<source>/aggregate.json`
contains per-sequence metrics and output paths. Under each sequence/algorithm,
inspect `mot.txt`, `metrics.json`, `upstream.manifest.json` and `experiment.json`.
All 22 held-out reference/core comparisons were checked to use identical detection,
ground-truth and frame-time SHA-256 hashes. The original ONNX probes require the
already prepared caches in `runs/detector_cache/<sequence>/detections.csv`.

Limitations: GMOT-40 is an insect-only custom split with assumed nominal FPS, so
discovery delay should be interpreted in frames or explicitly nominal time.
Native-bee images have processed ROI backgrounds/overlays and sampling gaps; its
oracle inputs are association diagnostics. ByteTrack's NumPy overlap adapter and
the pinned TrackEval fragmentation caveat are documented in
`third_party/README.md`. Small subsets and differing lifecycle semantics do not
support a state-of-the-art claim.
