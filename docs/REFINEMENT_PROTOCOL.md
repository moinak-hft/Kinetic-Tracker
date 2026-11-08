# Motion/scene refinement: predeclared comparisons and scope

This extension follows an already inspected v2 study. The old synthetic test and
GMOT insect-2/3 clips are now **reused exploratory development evidence**, not fresh
blind tests. The source and v2 executable are retained so that the refinement can
be compared with the actual prior implementation under identical output rules.

`configs/refinement_protocol.json` is the machine-readable plan. The runner does
not train a detector or download/execute external code. It evaluates association
and detector-call placement on declared cached detector inputs; none of these
quality runs demonstrates a new end-to-end detector, live FPS or a field system.

## Equal output policy and association ablations

The primary comparison explicitly uses `--output-policy active` for both the old
binary and the new one, with min-hits 1, high/low .35/.08, IoU .1, max age .75
seconds and prediction horizon .24 seconds. `--paired-output` also runs `all` into
separate folders; the policies are never mixed within an aggregate or selected
using test results. The earlier v2 study's original `all` outputs remain unchanged.
The choice of active output was informed by the preceding v2 diagnostic and is
not presented as a newly blinded output-policy experiment.

The declared modes are:

| Mode | Executable and interpretation |
|---|---|
| `baseline_v2` | Actual retained `work/kinetic_baseline_v2.exe`, two-stage association. New scene/risk/motion flags are not forwarded to this older executable. |
| `two_stage` | Current executable's unchanged association control. Metrics and exact MOT output hashes are compared against the old binary on matching inputs. |
| `motion_uncertainty` | Mahalanobis hard gating plus soft uncertainty cost; direction cost and endpoint velocity recovery disabled. |
| `motion_direction` | Direction cost with uncertainty hard gating retained, soft Mahalanobis weight zero and recovery disabled. This is not a pure direction-only gate. |
| `motion_full` | Uncertainty/direction costs and real-observation endpoint velocity recovery. It is not full OC-SORT observation-centric re-update. |
| `motion_full_gate25` | One predeclared alternative hard gate (25 instead of 13.2767), with the other full-mode knobs unchanged. |
| `motion_hybrid_iou` | Validation-stage amendment: high-IoU fallback, zero uncertainty/direction soft weights, no endpoint recovery; strict non-overlap guards remain. |
| `motion_hybrid_full` | Same bounded high-IoU fallback with .25/.1 soft weights and endpoint recovery enabled. |
| `motion_hybrid_no_recovery` | Same hybrid full costs with endpoint recovery disabled. |
| `recovery` | Final validation-stage amendment: preserve the two original IoU confidence passes, then cautiously match only remaining confirmed tracks with sufficient real history to unmatched high-score detections before birth. |

Weights are fixed at .25 Mahalanobis and .1 direction where enabled. The shared
motion guards are max center distance 3 diagonals, cost limit 1 and ambiguity
margin .03. The motion gate is a heuristic with this filter/noise model; it is not
claimed to be statistically calibrated. Ordinary real-detection Kalman correction
remains enabled even when endpoint recovery is disabled. There is no synthetic
measurement insertion, future trajectory access or appearance-model training.

## Validation-only selection and explicit freeze

Selection uses the six synthetic validation clips and the raw GMOT insect-1
validation clip. The objective is fixed **before scoring**:

```text
balanced HOTA = .5 * mean(HOTA on six synthetic validation clips)
              + .5 * HOTA on raw insect-1 validation
```

The same weights define the IDF1 tie-break. A motion candidate is eligible only
if its raw-insect validation HOTA is no more than .005 below the new `two_stage`
control. Exact ties prefer unchanged `two_stage`; if no acceptable motion method
improves the objective, it remains the selected method. The initial grid contains
the strict ablations and two full-mode gates. Strict motion candidates then
regressed on raw validation, so three bounded hybrid variants were added during
validation development, before any fresh confirmation score was inspected. Their
high-IoU fallback avoids applying a poorly calibrated uncertainty gate to an already
plausible overlap match, while non-overlap pairs retain strict uncertainty/ambiguity
guards. This is an explicit protocol amendment, not an originally blinded
hypothesis. The objective/guard are unchanged and strict failures remain reported.
The three hybrid candidates also failed the raw validation guard. One final
`recovery` candidate was therefore added before confirmation: it prevents weaker
non-overlap edges from displacing strong IoU matches by retaining both original
passes first, then uses a fixed 2D center Mahalanobis gate 9.2103 for cautious
unmatched recovery. Class, center-distance, size, direction and ambiguity bounds
remain; endpoint velocity is updated only for recovered pairs. No additional grid
is introduced, and the unchanged two-stage control remains the fallback if this
candidate fails validation. All these unsuccessful development rounds are retained.
Oracle bee data are excluded from this objective and stay diagnostic.

Because skip/miss recovery can be inactive in every-frame tracking, there is a
separate, declared **budget-profile selection** on `fixed2` validation. It uses
the same source weights, HOTA/IDF1 objective and .005 raw guard relative to
`two_stage` at fixed2. The primary every-frame quality selection is unchanged.
A budget schedule is then chosen only among the already declared settings on
validation under that budget profile. Root validation tooling records these
secondary decisions in `configs/recommended_profile.json`; no new numerical
parameters or confirmation-driven choice are introduced. Fresh comparisons may
include the quality-selected and budget-selected profiles plus old/new two-stage
controls, all declared before fresh scoring.

Commands from the project root:

```powershell
.\.venv\Scripts\python.exe tools\refinement_study.py --phase validation --output runs\refinement
# Inspect validation/selection.json and baseline regression checks before freezing.
.\.venv\Scripts\python.exe tools\refinement_study.py --phase freeze --output runs\refinement
.\.venv\Scripts\python.exe tools\refinement_study.py --phase exploratory --output runs\refinement
.\.venv\Scripts\python.exe tools\refinement_study.py --phase confirmation --output runs\refinement
```

`freeze.json` pins the complete protocol, validation selection, executable hashes,
runner/evaluator hashes and additional confirmation inputs. The freeze refuses
changed validation inputs or changed executables. Confirmation checks those pins
and cannot run through this tool without an explicit freeze. A frozen study is
not silently overwritten: changing methods requires a new declared study/output
folder and revised evidence labels. All listed ablations can be reported, but
the primary selected method is determined by validation, not by which method
wins on an additional clip.

Before any run directory is created or executable starts, generated MOT, telemetry,
frame/detection logs, summaries, metrics and experiment files are checked against
protected detector/GT/time/metadata/protocol inputs and relevant source images.
Resolved path aliases and existing hard links are rejected. Aggregate, selection
and freeze JSON writes receive corresponding protection. If present,
`configs/recommended_profile.json` is copied into the freeze with its SHA-256;
later confirmation rejects a changed recommendation. Fresh association comparisons
are restricted to the validation-selected method and the old/new two-stage
controls. The predeclared scene/schedule comparisons remain available as controlled
ablations, without changing the validation recommendation afterward.
Freeze and confirmation also stream the actual ordered source image bytes and
verify their SHA-256 against metadata, compare the file count to `num_frames`,
and reject escaped image paths, symlinks and non-file entries. A stat-invalidated
memo avoids rereading unchanged files within a process. Modifying a JPEG while
leaving its metadata unchanged therefore fails the confirmation integrity check;
the tool never needs to decode or execute image contents for this check.

## Additional category confirmation data

Two clips were declared before measuring tracking performance: **bird-2** and
**fish-2** from the authors' GMOT-40 v0.1 release. They contain 357 and 151 source
JPEGs respectively. They were previously unused in the v2 experiments. Selection
was for nine additional non-insect categories and manageable data size, not for
measured favorable results. They provide a limited additional confirmation check,
not an insect-specific improvement claim or verified recording-independent
generalization. Session/source independence beyond dataset sequence names remains
unverified.

```powershell
.\.venv\Scripts\python.exe tools\prepare_extra_data.py --download
```

The preparation tool reads the already pinned official ZIP central directory and
label archives, downloads only two declared image regions (about 73.6 MB), requires
exact HTTP 206 / Content-Range responses, checks original member CRC32 and records
range/JPEG SHA-256 hashes. It preserves JPEG bytes and original author annotation
files; no image is re-encoded and no downloaded script/model runs. Ground truth
is converted to MOT format, detector caches retain author confidence/geometry,
and source-frame mappings remain explicit. Conversion performs structural label
checks but **does not compute tracking scores**. A 300 MB declared transfer ceiling
and bounded range retries avoid fetching the entire 1.5 GB source archive.

Prepared paths are `data/gmot40_confirmation/bird-2` and `fish-2`, each with
`metadata.json`, `frames.csv`, `img1/`, `gt/gt.txt`, `detections.csv`, preserved
`source_annotations/` and per-image integrity records. Root `provenance.json`
records the release, range offsets, checksums, attribution and
[CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The source is [the authors' GMOT-40 dataset](https://spritea.github.io/GMOT40/download.html).
Author detections follow their one-shot first-frame exemplar protocol and are
not this project's original ONNX detector. Capture FPS is not supplied, so
timestamps use declared nominal 25 FPS and frame delays are the primary measure.

## Scene ablations and exact matched controls

The protocol separately declares legacy/robust scene methods crossed with all/
active scheduler risk scope at the same .5 detector-call cap. The old binary can
represent only legacy scene/all risk scope; unsupported treatments are skipped
explicitly, never silently downgraded. These options can be run on selected
declared profiles without changing association thresholds:

```powershell
.\.venv\Scripts\python.exe tools\refinement_study.py --phase validation --profiles two_stage motion_full --schedules every adaptive50_legacy_all adaptive50_legacy_active adaptive50_robust_all adaptive50_robust_active --output runs\scene_diagnostic
```

A partial-profile validation command is a diagnostic; it cannot produce a valid
selection/freeze until all eligible profiles and all declared validation cases
are evaluated. For any adaptive treatment, its exact realized call count `K` is
replayed on `N` frames using the declared uniform control. The runner clones
**all** association, motion, scene, risk-scope and output-policy knobs; it changes
only schedule to uniform and sets `uniform_calls=K`, then asserts both exact count
and identical retained knobs. This is retrospective matched-total-budget analysis,
not an online comparator with unknown final `K` or proof of equal prefix budgets.
Per-prefix hard budget compliance is checked on risk-based run logs. Saturation
fractions expose cases where risk/image triggers are almost always active and
the method collapses to periodic detection.

## Metrics, regressions and honest reporting

Each case retains the full command, input hashes, executable/evaluator/runner
hashes, MOT output, TrackEval metrics and per-frame schedule diagnostics. Reports
include HOTA, IDF1, AssA, ID switches, FP/FN, detector-call rate and detector-box
discovery. Discovery is first selected box at score at least .08 and IoU at least
.5 to a GT identity; it is not high-confidence new-track initialization. Timing
starts at annotated visibility, not verified physical birth. Native-bee oracle
boxes are association diagnostics, and its processed backgrounds/gaps remain a
limitation. Cached replay timing is not detector runtime.

Aggregate quality scores are unweighted sequence means unless the explicitly
source-balanced validation selection is being reported; neither is a pooled
official benchmark score. Primary active and diagnostic all outputs remain
separate. Every new category regression must be shown without retuning on it.
Failing to improve validation selects the unchanged control; failing additional
confirmation narrows the conclusion rather than being hidden by a new threshold.

## Expanded confirmation before freezing

At the user's request the original two additional clips were expanded **before
any confirmation scoring** to nine GMOT40 sequences: airplane-2, ball-3,
balloon-2, bird-2, boat-2, car-2, fish-2, person-0 and stock-1. This is one clip
from every non-insect category, 1,781 original frames. Index2 was used except
ball3/person0/stock1 to keep selective image downloads under300MB (actual205MB).
Selection was based on category coverage and archive size, not measured tracking
performance. Image ranges, original member CRCs, SHA256 and source licences are
recorded. No neural detector was trained or rerun for these cached comparisons.

The two operating profiles and all thresholds were frozen before scoring.
`baseline_v2`, `two_stage` and the validation-selected budget profile are run
on every/fixed2/fixed4 and the four declared scene/risk schedules. Unsupported
new scene/risk settings are omitted for the old executable, never silently
downgraded. Every adaptive run has a uniform control with exactly the same K
selected detections and cloned remaining knobs. One clip per category is broader
coverage, not proof of full-benchmark or recording-independent generalisation.

Official OC-SORT and ByteTrack references use the same source caches and fixed
settings in `reference_controls`. Their native track lifecycle/output differs
from the controlled same-output-policy comparisons; this is stated separately.
See `REFINEMENT_REPRODUCE.md` for the complete command order, including budget
selection before freeze, reference runs and the final report.
