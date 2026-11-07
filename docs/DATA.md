# Data, provenance and scope

The repository contains deliberately separate evaluation sources. Synthetic
results are diagnostics; Monash images are processed real recordings; and the
GMOT40 insect subset provides natural full-scene images with author-released
one-shot detector outputs. None is presented as a field deployment of a newly
trained agricultural-pest detector.

## GMOT40 insect subset: natural full-scene recordings

After visually finding ROI masks in the Monash release, we acquired an independent
full-scene source from the GMOT40 authors. The inspected held-out clips show ants
moving over ground (`insect-2`) and flying bees near a hive (`insect-3`). Source
JPEGs are preserved byte-for-byte. Objects vary in size; these are not uniformly
tiny objects. The ant video contains its original contributor watermark.

- Authors: Hexin Bai, Wensheng Cheng, Peng Chu, Kai Zhang and Haibin Ling.
- Paper: **GMOT-40: A Benchmark for Generic Multiple Object Tracking**, CVPR 2021.
- Author download page: https://spritea.github.io/GMOT40/download.html
- Official dataset release: https://github.com/Spritea/GMOT40/releases/tag/v0.1
- Dataset licence: **CC BY-NC-SA 4.0**. Preserve author attribution and use the same
  applicable licence for shared dataset-derived demo footage. The repository's
  source-code licence does not relicense the dataset.

Reproduce the selective data preparation:

```powershell
.\.venv\Scripts\python.exe tools/prepare_data.py --gmot --download
```

The full image archive is 1,498,450,771 bytes. This tool downloads its pinned central
directory and only the contiguous insect image region, bytes 876055443 through
990099285: **114,043,843 image-range bytes**, rather than the entire archive. It
requires exact HTTP 206/Content-Range responses and pinned SHA-256 values. The two
small author GT/detection archives are also pinned. ZIP CRCs and JPEG dimensions
are verified while reading selected entries; downloaded content is never executed.
All source URLs, byte ranges and hashes appear in `data/gmot40/provenance.json`.

| Role | Sequences | Images |
| --- | --- | ---: |
| Train / development | insect-0 | 110 |
| Validation | insect-1 | 201 |
| Held-out test | insect-2, insect-3 | 348 |

This is a declared **insect-only custom subset protocol**, not the complete official
40-sequence GMOT40 benchmark. Sequences are disjoint; underlying source-session
independence beyond sequence IDs is not verified. The image archive lacks verified
capture timestamps or FPS. `frames.csv` therefore uses observation index / nominal
25 FPS, explicitly marked as an assumption. Report discovery delay in **frames**;
do not infer physical flight speed or original-camera real-time deadlines from it.

`detections.csv` comes from the authors' released **one-shot detector**, which uses
a target exemplar from each sequence's first frame under their protocol. It is not
output from Kinetic-Tracker's `best.onnx`, not oracle GT, and not a newly trained
detector result. The scheduler must decide from pixels and past state before
reading the current cached detections. These experiments test scheduling and
association on a real, fixed detection stream. They do not measure end-to-end
neural-inference throughput. `oracle_detections.csv` is separately identified and
only tests association in isolation.

## Monash native-bee data

- Dataset: **Australian Native Bee Detection and Tracking Dataset**, Asad Asaduzzaman,
  Malika Nisal Ratnayake, Alan Dorin and Vanessa Kellermann, Monash University (2026).
- DOI: https://doi.org/10.26180/32315067.v1
- University record: https://bridges.monash.edu/articles/dataset/Australian_Native_Bee_Detection_and_Tracking_Dataset/32315067
- Author code: https://github.com/asadiceiu/native-bee-detection-tracking
  (inspected commit `9589ef2c88d0b8e85112550519185a02a21df9c3`).
- Dataset licence: **CC BY 4.0**, as recorded in the Figshare API. Preserve attribution
  when redistributing images. The author's code licence is separate.
- Tracking archive: https://ndownloader.figshare.com/files/64894986
- Bytes: `107325128`; provider MD5: `9c87d57716c879c2b569196ba4197021`.
- SHA-256: `fb79d8bb77cdedd394713d56396755314fa6446ecc2795dbce62a9da3109cd8b`.

Only data files are downloaded. The preparation script restricts HTTPS redirects
to Figshare/Monash hosts, checks pinned hashes and sizes, rejects archive traversal
and symbolic links, validates JPEG dimensions, and never executes archive content.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe tools/prepare_data.py --download --export-yolo
```

The published archive has 16 sequences. Three have fewer than 20 supplied images
(6, 9 and 19 images), so they are excluded before experiments. Of the remaining
3,511 images, source frame 219 of `Camera_43_20240423_124240` assigns the identity
`bee_31` to two different boxes. The preparation script quarantines this entire
ambiguous frame instead of inventing an identity or scoring contradictory labels.
The prepared benchmark therefore has **13 sequences and 3,510 scored images**.
The quarantine rule applies before metric calculation and is not selected by results.
Its exact source member and reason are recorded in `data/native_bee/provenance.json`.

### Frozen split

All recordings from a calendar date stay together. The split is declared in code
and written to `configs/splits.json` before tracking evaluation.

| Role | Recording dates | Sequences | Supplied valid images |
| --- | --- | ---: | ---: |
| Train | 2024-04-11, 2024-04-12, 2024-04-23 | 6 | 1,496 |
| Validation | 2024-05-02, 2024-05-31 | 4 | 497 |
| Test | 2024-04-20, 2024-05-05 | 3 | 1,517 |

This is a same-camera/same-hive study across recording days. It does not establish
cross-camera, cross-species, new-site, or agricultural-pest generalisation.
The detector-export option converts identity boxes to one `insect` detector class;
identity labels are retained only for tracking evaluation. Detector training,
parameter selection and final evaluation must respect these same day groups.

### Important processed-image limitation

Visual inspection found that supplied JPEGs contain **black-masked backgrounds,
ROI patches around visible insects, and embedded green frame/time counters**.
They are processed real images, not unmodified full-scene camera frames. The moving
ROI boundaries or changing counters can supply artificial scene-change cues.

Consequently, oracle association tests and a detector-transfer probe are useful,
but scheduler improvements on this source must not be presented as demonstrated
new-arrival discovery on natural raw footage. Masking the counter strip alone
does not remove the ROI-boundary confound. Raw, unprocessed held-out recordings
are required for that deployment claim.

Filenames also have missing source indices. The converter numbers available images
contiguously for MOT output but preserves original indices in `frames.csv`.
The published nominal rate is 25 FPS, so the timestamp column is
`(source_frame - first_available_source_frame) / 25`. This is **nominal dataset
timing**, not independently verified sensor capture timing. Embedded raw-frame/time
counters suggest an additional upstream processing timeline. Missing source images
are not fabricated or scored. Algorithms using real elapsed motion must consume
the timestamp map and disclose this uncertainty; fixed-step reference trackers are
evaluated over the supplied observation sequence.

### Files per prepared sequence

```text
img1/000001.jpg             Original supplied JPEG, byte-preserving reindex
gt/gt.txt                  frame,id,x,y,w,h,1,1,visibility (MOT format)
frames.csv                 frame,timestamp,source_frame,source_member
metadata.json              Geometry, counts, split, IDs and data limitations
seqinfo.ini                MOT sequence geometry and nominal FPS
oracle_detections.csv      Perfect GT boxes without identities, score=1
```

`oracle_detections.csv` is **not a neural detector result**. It isolates association
behaviour using per-current-frame ground-truth boxes. It must never be read on
skipped frames to decide whether a detection should run. The original project's
`best.onnx` training set is unknown; its scores do not establish clean held-out
detector generalisation. Any later known-provenance detector should record its
training list, weights hash, model configuration and package versions.

## Synthetic diagnostics

Generate the deterministic diagnostics:

```powershell
.\.venv\Scripts\python.exe tools/generate_synthetic.py
```

There are six scenario families, each with independent validation and test seeds:
smooth motion, crossings, abrupt turns, temporary occlusion, short-lived arrivals,
and moving background shadows. Defaults are 180 frames per sequence, 640x360 at
25 FPS, for 12 sequences and 2,160 rendered frames.

The generator writes rendered JPEG frames, MOT ground truth, causal timestamps,
and noisy synthetic detection caches with lower confidence near difficult events,
misses and occasional false positives. These caches are simulated observations,
not YOLO outputs. Object identities are not included in the detector cache.
Fully hidden objects are unscored while hidden and resume their previous identity
after visibility returns. Detector skipping adds further missing observations.

The scheduler may examine the current rendered frame and past tracker state.
It must not examine the current cached detections or labels until it has decided
to run the detector. Pixel-derived cue extraction overhead belongs in deployment
timings. Cached replay measures algorithm behaviour, not neural-inference speed.

The validation/test split is new random seeds **within the same scenario families**.
It is not evidence of unseen-family or real-world generalisation. Seed values,
generator version, image hash, label hash and detection hash are in each sequence's
metadata and the synthetic provenance manifest.
