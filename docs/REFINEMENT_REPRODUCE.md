# Reproduce the refinement

The current checkout is `D:\Kinetic-Tracker-main` on
`research/motion-aware-tracking`. The folder name does not determine its Git
branch. Old E: folders are separate historical copies.

## Build and run locally

Python and the MSYS2 UCRT64/OpenCV toolchain are already installed on this PC.
The project virtual environment is isolated. No PowerShell activation or
execution-policy change is needed:

```powershell
Set-Location 'D:\Kinetic-Tracker-main'
git branch --show-current
.\.venv\Scripts\python.exe tools\build.py --jobs 2
.\.venv\Scripts\python.exe tools\run_profile.py --frames 240
.\.venv\Scripts\python.exe tools\run_profile.py --budget --frames 240
Start-Process '.\results\refinement\index.html'
```

The first run uses the validation-selected quality profile. `--budget` uses
the separately selected lower-call option; its trade-offs are in the report.
Outputs go to distinct timestamped `runs/demo_*` folders. For another video:

```powershell
.\.venv\Scripts\python.exe tools\run_profile.py --video 'D:\path\clip.mp4' --model '.\best.onnx' --output 'runs\my_clip'
```

`best.onnx` is the original single-class pest detector, not a general detector
for all nine additional categories. Those controlled tracking experiments use
the dataset authors' fixed detections. `yolov8n.onnx` is retained as a separate
pretrained model; changing model changes the detector task and its evidence.

On a fresh checkout, create `.venv` with Python and install the pinned packages
from official PyPI before building. The toolchain must provide CMake, Ninja,
g++ and OpenCV; `tools/build.py --toolchain-root <UCRT64-root>` selects it.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -r requirements-lock.txt
.\.venv\Scripts\python.exe tools\build.py --jobs 2
```

## Scientific reproduction

The checked-in report contains compact measured evidence. Large data, virtual
environments, executables, raw runs and temporary downloads are ignored by Git.
The local D: copy has the prepared data. `docs/REPRODUCE.md` documents v2 data
preparation; `tools/prepare_extra_data.py --download` adds nine confirmation
clips using bounded HTTP206 ranges from the official GMOT40 GitHub release.
No downloaded code is executed by that data tool. Original JPEG CRCs and archive,
image, label and range hashes are retained in data provenance JSON.

Use a **new** output folder to repeat the study. Existing `runs/refinement`
contains the actual frozen study; do not overwrite its selection or rebuild its
executable midway through confirmation. Rebuilding from identical source can
change binary hashes through compiler/linker timestamps. No post-confirmation
retuning is part of this study.

```powershell
.\.venv\Scripts\python.exe tools\prepare_baseline.py
.\.venv\Scripts\python.exe tools\prepare_extra_data.py --download
.\.venv\Scripts\python.exe tools\refinement_study.py --phase validation --output runs\refinement_repeat
.\.venv\Scripts\python.exe tools\refinement_schedule.py --study runs\refinement_repeat
.\.venv\Scripts\python.exe tools\refinement_study.py --phase freeze --output runs\refinement_repeat
.\.venv\Scripts\python.exe tools\refinement_study.py --phase confirmation --output runs\refinement_repeat --schedules every fixed2 fixed4 adaptive50_legacy_all adaptive50_legacy_active adaptive50_robust_all adaptive50_robust_active
.\.venv\Scripts\python.exe tools\refinement_references.py --runs runs\refinement_repeat
.\.venv\Scripts\python.exe tools\refinement_study.py --phase exploratory --output runs\refinement_repeat --profiles baseline_v2 two_stage motion_hybrid_full --schedules every fixed2 adaptive50_legacy_all adaptive50_legacy_active adaptive50_robust_all adaptive50_robust_active
.\.venv\Scripts\python.exe tools\refinement_study.py --phase diagnostic --output runs\refinement_repeat --profiles baseline_v2 two_stage motion_hybrid_full --schedules every
.\.venv\Scripts\python.exe tools\refinement_runtime.py --runs runs\refinement_repeat
.\.venv\Scripts\python.exe tools\refinement_demos.py --runs runs\refinement_repeat --output results\refinement_repeat --group confirmation
.\.venv\Scripts\python.exe tools\refinement_demos.py --runs runs\refinement_repeat --output results\refinement_repeat --group exploratory
.\.venv\Scripts\python.exe tools\refinement_report.py --runs runs\refinement_repeat --output results\refinement_repeat
```

The baseline comes from merged commit
`e2d2343fd3814d1f4fe114172373d801ed7ca7f8`, built with the same compiler/OpenCV.
Reference controls are pinned official tracking subsets, with explicit native
lifecycle semantics and compatibility adapters. Their fixed settings are in the
frozen protocol. Run timing after other compute jobs finish; it uses three
counterbalanced repetitions and excluded warmups on the original unlabelled
240-frame video. It is not a tracking-accuracy benchmark.

The reporter does not select winners. It checks frozen records, matching inputs,
executable hashes and exact-K controls, then derives HTML, plots, Markdown and
CSV/JSON exports. Quality metrics use normalized scores internally and 0–100
points in figures. Reported clip means are not official pooled benchmark scores.
