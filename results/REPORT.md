# Kinetic-Tracker: measured implementation report

Native C++17 tracking, detector-budget scheduling, and reproducible MOT evaluation. Results are exploratory evidence on a small subset, not a state-of-the-art or field-deployment claim.

## Main findings

- The confidence-based two-stage association improves over the corrected SORT-style baseline on the tested natural ant/bee clips using the exact same author-provided detections.
- Adaptive scheduling modestly improves over uniform selection at exactly matched detector-call counts on the synthetic test suite. On natural clips it collapses to periodic detection and does not beat the rate-matched baseline.
- Original pest-model transfer is weak. Stronger association does not substitute for a well-trained detector.
- Output policy matters: stale predictions can increase false positives. Both policies and their timing of introduction are disclosed.

## Protocol

Primary frozen scheduler configurations were selected on six synthetic VALIDATION clips only, then evaluated on separate synthetic seeds and real sequence splits. The original study deliberately retains all non-tentative estimates (`--output-policy all`). Scores below are unweighted means over sequences, not pooled official benchmark scores. HOTA/IDF1 are shown as percentages; error counts are summed. Exact matched baselines use each adaptive method's realized per-sequence call count, with content-independent evenly spaced positions.

Corrected SORT and two-stage are local implementations. ByteTrack and OC-SORT are pinned official tracking-only implementations; ByteTrack uses documented compatibility adapters. Upstream output rules, threshold details and fixed-step timing differences are retained and recorded.

## Synthetic diagnostics: 6 clips, 1,080 frames

Generated frames and noisy simulated detections. Different seeds within the same six scenario families.

| Method | Detector calls | HOTA | IDF1 | ID switches | FP | FN |
|---|---|---|---|---|---|---|
| Corrected SORT | 100.0% | 76.53 | 78.80 | 20 | 299 | 109 |
| Two-stage | 100.0% | 80.16 | 84.38 | 15 | 311 | 102 |
| Fixed 1/2 | 50.0% | 77.05 | 82.15 | 21 | 244 | 85 |
| Fixed 1/4 | 25.0% | 68.75 | 75.40 | 32 | 310 | 305 |
| Fixed 1/8 | 12.8% | 36.19 | 43.58 | 196 | 883 | 1870 |
| Covariance, cap50% | 23.5% | 60.88 | 73.27 | 23 | 188 | 743 |
| Risk, cap50% | 23.8% | 60.72 | 73.12 | 23 | 188 | 753 |
| Adaptive, cap25% | 22.2% | 68.35 | 75.54 | 31 | 307 | 294 |
| Adaptive, cap50% | 34.1% | 71.96 | 78.08 | 19 | 222 | 142 |
| Uniform matched25 | 22.2% | 66.75 | 74.77 | 33 | 304 | 371 |
| Uniform matched50 | 34.1% | 70.63 | 76.39 | 24 | 259 | 202 |
| Official OC-SORT | 100.0% | 68.01 | 78.69 | 8 | 0 | 522 |
| ByteTrack + adapters | 100.0% | 75.56 | 82.74 | 14 | 0 | 288 |

## Natural insect footage: 2 clips, 348 frames

Author-provided one-shot detector outputs on untouched GMOT-40 ant/bee frames. This is cached association/scheduling evaluation, not accuracy from our ONNX model.

| Method | Detector calls | HOTA | IDF1 | ID switches | FP | FN |
|---|---|---|---|---|---|---|
| Corrected SORT | 100.0% | 26.25 | 35.89 | 179 | 1665 | 2116 |
| Two-stage | 100.0% | 29.30 | 41.36 | 175 | 1846 | 1924 |
| Fixed 1/2 | 50.3% | 24.52 | 32.68 | 253 | 2221 | 2264 |
| Fixed 1/4 | 25.3% | 18.30 | 21.33 | 184 | 1780 | 2807 |
| Fixed 1/8 | 12.6% | 9.33 | 9.29 | 137 | 1269 | 3314 |
| Covariance, cap50% | 46.8% | 22.80 | 28.14 | 252 | 2217 | 2388 |
| Risk, cap50% | 49.1% | 23.53 | 30.45 | 250 | 2173 | 2324 |
| Adaptive, cap25% | 25.3% | 18.30 | 21.33 | 184 | 1780 | 2807 |
| Adaptive, cap50% | 50.3% | 24.52 | 32.68 | 253 | 2221 | 2264 |
| Uniform matched25 | 25.3% | 18.92 | 22.26 | 175 | 1727 | 2784 |
| Uniform matched50 | 50.3% | 24.52 | 32.68 | 253 | 2221 | 2264 |
| Official OC-SORT | 100.0% | 27.07 | 39.24 | 119 | 187 | 2380 |
| ByteTrack + adapters | 100.0% | 23.43 | 32.72 | 77 | 237 | 2716 |

## Processed bee recordings: oracle association diagnostic

Ground-truth boxes are used as perfect detections. ROI masks/counters and sampling gaps prevent natural scene/new-arrival deployment claims.

| Method | Detector calls | HOTA | IDF1 | ID switches | FP | FN |
|---|---|---|---|---|---|---|
| Corrected SORT | 100.0% | 57.48 | 69.22 | 83 | 1236 | 6 |
| Two-stage | 100.0% | 57.48 | 69.22 | 83 | 1236 | 6 |
| Fixed 1/2 | 50.0% | 55.74 | 71.45 | 55 | 954 | 143 |
| Fixed 1/4 | 25.0% | 51.08 | 67.88 | 39 | 708 | 351 |
| Fixed 1/8 | 12.5% | 39.04 | 56.02 | 40 | 477 | 860 |
| Covariance, cap50% | 28.9% | 43.09 | 59.71 | 68 | 818 | 507 |
| Risk, cap50% | 28.9% | 43.09 | 59.71 | 68 | 818 | 507 |
| Adaptive, cap25% | 25.0% | 51.08 | 67.88 | 39 | 708 | 351 |
| Adaptive, cap50% | 50.0% | 55.74 | 71.45 | 55 | 954 | 143 |
| Uniform matched25 | 25.0% | 51.69 | 68.91 | 36 | 721 | 358 |
| Uniform matched50 | 50.0% | 55.73 | 71.45 | 55 | 954 | 143 |
| Official OC-SORT | 100.0% | 93.94 | 93.84 | 15 | 0 | 108 |
| ByteTrack + adapters | 100.0% | 79.24 | 91.20 | 13 | 1 | 172 |

## Original pest ONNX model: transfer probe

Actual OpenCV DNN detections on natural held-out ant/bee clips. Unknown original training provenance; this is a transfer probe, not a newly trained detector.

| Method | Detector calls | HOTA | IDF1 | ID switches | FP | FN |
|---|---|---|---|---|---|---|
| Corrected SORT | 100.0% | 7.76 | 9.09 | 35 | 587 | 3544 |
| Two-stage | 100.0% | 10.40 | 12.71 | 44 | 919 | 3331 |
| Fixed 1/2 | 50.3% | 7.94 | 9.28 | 39 | 740 | 3481 |
| Fixed 1/4 | 25.3% | 8.10 | 9.05 | 15 | 388 | 3604 |
| Fixed 1/8 | 12.6% | 4.35 | 4.57 | 9 | 199 | 3724 |
| Covariance, cap50% | 39.1% | 8.88 | 10.19 | 43 | 740 | 3482 |
| Risk, cap50% | 39.4% | 8.63 | 9.94 | 40 | 716 | 3492 |
| Adaptive, cap25% | 25.3% | 8.10 | 9.05 | 15 | 388 | 3604 |
| Adaptive, cap50% | 50.3% | 7.94 | 9.28 | 39 | 740 | 3481 |
| Uniform matched25 | 25.3% | 7.38 | 8.34 | 16 | 413 | 3613 |
| Uniform matched50 | 50.3% | 7.94 | 9.28 | 39 | 740 | 3481 |

## Post-hoc output policy: natural clips

**Exploratory post-hoc diagnostic:** investigated after inspecting first-pass test false positives. It was then checked on validation, and the CLI default changed to `active`. These numbers are not an untouched confirmatory test. Primary frozen-study scores above retain `all`. Telemetry preserves all estimates in both modes; active suppresses detector-confirmed lost boxes from scored/rendered output but retains predictions during deliberate skips.

| Method | Detector calls | HOTA | IDF1 | ID switches | FP | FN |
|---|---|---|---|---|---|---|
| Two-stage active output | 100.0% | 30.85 | 44.66 | 177 | 573 | 2065 |
| Two-stage all estimates | 100.0% | 29.30 | 41.36 | 175 | 1846 | 1924 |

## Post-hoc output policy: oracle bee boxes

**Exploratory post-hoc diagnostic:** investigated after inspecting first-pass test false positives. It was then checked on validation, and the CLI default changed to `active`. These numbers are not an untouched confirmatory test. Primary frozen-study scores above retain `all`. Telemetry preserves all estimates in both modes; active suppresses detector-confirmed lost boxes from scored/rendered output but retains predictions during deliberate skips.

| Method | Detector calls | HOTA | IDF1 | ID switches | FP | FN |
|---|---|---|---|---|---|---|
| Two-stage active output | 100.0% | 81.84 | 90.48 | 83 | 6 | 6 |
| Two-stage all estimates | 100.0% | 57.48 | 69.22 | 83 | 1236 | 6 |

## Actual uncached CPU timing

Three counterbalanced repetitions on the original unlabelled sample, 8 OpenCV CPU threads. Baseline mean wall time: 24.391s; adaptive mean wall time: 11.168s. Wall-time speedup: 2.18x. This input has no identity labels; the timing result cannot establish accuracy. Initialisation and one warmup inference are separate. No GPU or energy claim.

## Limits and next experiments

- Natural scheduling evidence is only two GMOT-40 insect clips; the detector is the author's one-shot baseline and uses its target exemplar protocol. This is a custom subset, not an official full benchmark submission.
- Real capture timing is not verified. Report discovery delays in frames; nominal millisecond conversions are not physical measurements.
- Monash release images have annotation-related ROI masks and counters. Oracle detections isolate association and are never represented as detector performance.
- Original `best.onnx` training data/code are absent; its evaluation is a transfer probe with unknown training provenance.
- Adaptive foreground novelty is sensitive to vegetation, camera motion, missed tracks and small targets. Current evidence supports a prototype and a negative real-data scheduling result.
- The active-output change is a post-hoc diagnostic. A new independent raw recording is needed to confirm that decision without development-set reuse.
- Pinned TrackEval CLEAR Frag may undercount wholly empty prediction gaps; HOTA/IDF1 and FP/FN are the primary metrics.
- Next: acquire independent raw labelled pest videos; establish detector recall with documented training provenance; compare a learned/size-aware motion gate against OC-SORT; test scene-motion rejection at matched budgets.

## Reproduction and attribution

See `docs/REPRODUCE.md`, `docs/METHOD.md`, `docs/ARCHITECTURE.md`, `docs/DATA.md`, `docs/RELATED_WORK.md` and pinned `third_party/PROVENANCE.json`. Raw inputs/caches are ignored by Git. Configurations, summary tables, report and licensed demo attribution are tracked.

GMOT-40: Bai et al., CVPR2021, https://github.com/Spritea/GMOT40 (dataset CC BY-NC-SA4.0). Monash native bees: DOI10.26180/32315067.v1 (CC BY4.0). Metrics: https://github.com/JonathonLuiten/TrackEval. References: https://github.com/noahcao/OC_SORT and https://github.com/FoundationVision/ByteTrack. Scheduling prior work: DriftSched, DOI10.1109/MECO70748.2026.11579050.
