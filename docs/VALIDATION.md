# Verification of the completed refinement

Verified locally on 8 November 2026. Source is on
`research/motion-aware-tracking`; no remote push or merge is included.

## Software checks

- Windows 11, Python 3.14.4, GCC 15.2.0 UCRT64, OpenCV 4.12 C++ backend.
- Release C++ build completed using the existing local toolchain.
- **Six CTest groups pass:** tracker, scheduler, detector, motion, scene and
  Python integration/evaluation tests.
- **81 Python tests run: 80 passed, 1 skipped** for unavailable Windows symlink
  privileges. Input/output alias protection also has hard-link checks.
- Pinned GitHub Actions CPU CI is configured. The cloud workflow has not been
  run by this local task; local success is not presented as cloud CI success.
- The frozen protocol, selection, source images and both executables pass
  integrity checks after all confirmation runs.

## Experimental checks

- Validation: seven cases, all declared association candidates retained, with
  source-balanced scoring and the raw-insect guard. Quality and reduced-call
  recommendations were selected separately, then frozen.
- Additional confirmation: **nine non-insect category clips, 1,781 frames,
  243 controlled replay records**. No confirmation-driven retuning.
- The old executable and the new executable's legacy two-stage mode produce
  **byte-identical MOT outputs in 45 paired confirmation runs**: every/fixed2/
  fixed4/adaptive legacy and its exact-count uniform control across all 9 clips.
  The seven validation byte-regression checks also pass.
- The report derives 647 active-output experiment records across all phases,
  with 96 exact-count scheduling pairs and 18 official reference runs.
- Exact-count comparisons assert equal per-case calls and all other tracker,
  scene, risk and output knobs. Causal budget prefix limits are checked.
- Every source frame is retained in **11 comparison videos**; all output frame
  counts were decoded/verified. Nine cover the additional categories, one is
  earlier insect footage, and one is a synthetic background-motion diagnostic.
- Three excluded-warmup, counterbalanced uncached CPU runs per operating
  profile use the first 240 frames of the original unlabelled video. No competing
  project benchmark/rendering jobs were running during these measurements.

## Measured outcomes and scope

At fixed2, mean HOTA across nine additional clips is 35.82 for v2 and 36.55 for
hybrid motion with identical 893 detector calls. Six clips improve; three regress.
The validation-selected adaptive policy scores 36.53 at 860 calls, below its
same-association, exact 860 call uniform control's 36.66. Neither failure nor the
more favourable clips are removed.

Mean wall CPU FPS on the separate original video is 10.35 (v2), 10.33 (quality)
and 20.51 (budget); budget uses 112 rather than 240 detector calls, a 1.98× mean wall
FPS ratio. Three-run standard deviations are 0.34, 0.40 and 0.57 respectively.
These measurements do not establish processing speed on all additional clips,
tracking quality on the unlabelled video, or real-world capture timing.

Data preparation uses original author JPEGs, released labels/detections,
bounded HTTP206 ranges and recorded SHA256/CRC checks. No new detector training
or weight modifications were performed. See the system card for limitations,
the scientific protocol for selection rules and the report for all measured rows.
