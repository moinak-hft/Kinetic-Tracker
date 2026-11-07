# Reproduce the branch

All commands run from the repository root. Raw data, the Python virtual
environment, build products and detailed runs stay local and are ignored by Git.
Source, pinned third-party tracking code, configurations and compact result
artifacts are versioned. Existing root `main.cpp` and its old notebook are kept
as historical inputs; the new executable is built from `src/`.

## Tested environment

Windows, MSYS2 UCRT64 GCC15.2, OpenCV4.12 C++ libraries, Python3.14.4. Exact Python
packages are in `requirements-lock.txt`; machine/backend details are in
`results/environment.json`. Existing CPU/OpenCV tools were used. The RTX4060 is
listed for transparency and was not used for these measurements.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -r requirements-lock.txt
.\scripts\build.ps1
```

The build helper locates the tested `D:\Tools\msys64\ucrt64` installation, or
accepts `-ToolchainRoot C:\msys64\ucrt64`. It changes PATH only inside its process.
It does not install system software or alter execution/security policies.
OpenCV's C++ SDK and a compiler are prerequisites on a different computer.
On Linux use CMake with your OpenCV development package and set
`-DPython3_EXECUTABLE=/absolute/path/to/.venv/bin/python` for Python CTest checks.

## Run actual ONNX inference

```powershell
.\scripts\run.ps1 --video test_video.mp4 --model best.onnx --association two_stage --threads 8 --output runs\my_demo --output-video runs\my_demo\annotated.mp4
```

`--preview` is optional. No graphical window opens in the default headless mode.
The CLI defaults to `--output-policy active`; lost states remain in telemetry,
but a detector-confirmed missed box is not rendered/scored. Intentional skipped
frame predictions remain distinguishable through `observed=0`. `all` exports
all non-tentative estimates within the prediction horizon. Confidence is the
last detector score, not a calibrated track probability.

Inputs are never automatically renamed or deleted. Canonical collisions between
input assets and generated files are rejected. Choose a new output directory
when preserving an earlier run; existing generated files in a chosen run folder
are replaced.

## Acquire exactly the prepared data

```powershell
.\.venv\Scripts\python.exe tools\generate_synthetic.py
.\.venv\Scripts\python.exe tools\prepare_data.py --download --export-yolo
.\.venv\Scripts\python.exe tools\prepare_data.py --gmot --download
```

`prepare_data.py` fetches pinned data from official Monash/Figshare and GMOT-40
author sources. The GMOT loader uses verified HTTP byte ranges to retrieve only
the insect JPEGs instead of the full1.5GB archive. Dataset checksums, archive
validation, path traversal checks, annotation quarantine, licence and source
details are recorded in `docs/DATA.md` and `data/*/provenance.json`.

## Frozen primary study

```powershell
# Optional: reproduce selection using VALIDATION, never test.
.\.venv\Scripts\python.exe tools\tune_scheduler.py

.\.venv\Scripts\python.exe tools\benchmark.py --dataset synthetic --split test --configs configs\study.json --matched --output runs\synthetic_test
.\.venv\Scripts\python.exe tools\benchmark.py --dataset gmot40 --split test --configs configs\study.json --matched --output runs\gmot40_test
.\.venv\Scripts\python.exe tools\benchmark.py --dataset native_bee --split test --configs configs\study.json --matched --output runs\native_bee_oracle_test

.\.venv\Scripts\python.exe tools\reference_benchmark.py --dataset synthetic --split test --source provided
.\.venv\Scripts\python.exe tools\reference_benchmark.py --dataset gmot40 --split test --source provided
.\.venv\Scripts\python.exe tools\reference_benchmark.py --dataset native_bee --split test --source provided
```

The primary study explicitly passes `--output-policy all`, preserving the first
frozen protocol even though the interactive CLI now defaults to active output.
Matched uniform controls use the same final call count per sequence, chosen
retrospectively without ground-truth quality; they are not a prefix-identical
online policy. The headline HOTA/IDF1 summaries are sequence means, not pooled
official benchmark scores. Read `docs/METHOD.md` for all protocol distinctions.

## Test the original detector's transfer

```powershell
.\.venv\Scripts\python.exe tools\cache_detector.py --dataset gmot40 --split test
.\.venv\Scripts\python.exe tools\benchmark.py --dataset gmot40 --split test --source onnx --configs configs\study.json --matched --output runs\gmot40_onnx_test
```

This runs actual ONNX detection, then replays its cache for controlled tracking
comparisons. Original-model training provenance is missing. The supplied model
must not be represented as a newly trained or general insect detector.

## Output-policy diagnostic

```powershell
.\.venv\Scripts\python.exe tools\benchmark.py --dataset gmot40 --split test --configs configs\output_policy_diagnostic.json --output runs\output_policy_gmot_test
.\.venv\Scripts\python.exe tools\benchmark.py --dataset native_bee --split test --configs configs\output_policy_diagnostic.json --output runs\output_policy_native_test
```

This diagnostic followed inspection of first-pass held-out errors and is
explicitly exploratory. The active policy was checked on validation before
becoming the default. These numbers are kept separate from the primary frozen
scheduler protocol, and a new independent recording is required for confirmation.

## Actual timing, videos and report

```powershell
.\.venv\Scripts\python.exe tools\runtime.py --threads 8 --repeats 3 --frames 240
.\.venv\Scripts\python.exe tools\make_demo.py --sequence data\gmot40\insect-3 --left runs\gmot40_test\insect-3\every_sort --right runs\gmot40_test\insect-3\every_two_stage --left-label "Corrected SORT" --right-label "Two-stage association" --output results\gmot_comparison.mp4 --panel-width 800
.\.venv\Scripts\python.exe tools\report.py
```

The demo tool rejects an existing output to prevent accidental loss. Use a new
output filename when repeating it. Derived GMOT footage retains attribution and
its dataset licence. Playback25FPS is nominal, not verified source timing.
Runtime uses actual uncached inference on the original unlabelled clip,
counterbalanced over3 repeats. It cannot establish tracking accuracy on that clip.
Open `results/report.html` locally; it requires no server or external scripts.
