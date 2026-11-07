# Pinned reference implementations

`PROVENANCE.json` records the exact official repository commits and the limited
compatibility edits. This folder contains source subsets, not separate Git clones.
Every upstream license is preserved in its component folder. Small package marker
files avoid importing unrelated detectors and heavyweight training packages.

- **TrackEval:** official HOTA, Identity and CLEAR metric modules. Replaced removed
  NumPy `np.float` / `np.int` aliases with Python built-in types. The evaluation
  input adapter in `tools/evaluate.py` is project code, not an official dataset loader.
- **OC-SORT:** official tracker, association and Kalman implementation, without its
  detector/training dependencies. Association chooses the upstream LAP implementation
  when `lap` is installed, or its upstream SciPy fallback otherwise. Exact ties may
  be resolved differently by these two solvers; manifests record dependencies.
- **ByteTrack:** official tracker, Kalman filter, base classes and matching module.
  Removed two unused PyTorch imports because only the original NumPy Nx5 input path
  is used. Replaced removed NumPy scalar aliases. The optional tensor input branches
  are not supported by this subset. The original `lap.lapjv` thresholded assignment
  remains unchanged.
- **cython_bbox compatibility:** the compiled extension has no compatible wheel in
  the tested Windows/Python 3.14 environment. `ByteTrack/cython_bbox.py` provides a
  NumPy implementation of the upstream `bbox_overlaps` formula, including its
  **inclusive pixel (+1)** convention. It is tested against a literal scalar form
  of the upstream loop on random valid boxes and empty input. This convention differs
  from the continuous-box IoU used by TrackEval and our tracker; it is retained to
  preserve official ByteTrack costs. Original notices are in `CYTHON_BBOX_LICENSE`.

## Evaluation scope

The adapter scores class-1 insect MOT annotations, not the MOTChallenge pedestrian
distractor/ignore-region protocol. It rejects other marked classes rather than
silently excluding them. It removes unmarked ground truth and scores every valid
prediction without confidence filtering. HOTA/AssA/DetA/IDF1 are fractions; FP/FN
and IDSW are counts. CLEAR and Identity use IoU 0.5 by default; HOTA retains the
official 0.05 to 0.95 threshold sweep.

Known pinned-upstream limitation: CLEAR's early return for an entirely empty
prediction frame does not reset its fragmentation state. Consequently `Frag` may
undercount interruptions involving wholly empty prediction frames. We expose the
official result with that caveat instead of silently changing its definition.
The regression test explicitly covers this case. HOTA and IDF1 are unaffected by
this particular implementation issue.

## Reference replay scope

Use `tools/run_reference.py --algorithm ocsort` or `--algorithm bytetrack` with a
cached detector CSV and explicit source frame count. Empty frames still call the
upstream tracker update. Both references retain their original fixed time step:
the frame/timestamp CSV records source sampling gaps but does not modify the
reference Kalman model. In particular, sparse annotated images are not a continuous
25-FPS deployment demonstration. Output preserves the upstream active-track rules;
the wrapper does not add interpolated or lost-track boxes. OC-SORT's output score
1 is a MOT-format placeholder, not a probability.

Replay update timings exclude the detector, decoding and I/O. They cannot support
an end-to-end speedup claim. The ByteTrack reference is labeled **official ByteTrack
with compatibility adapters**, not an unmodified turnkey upstream environment.
