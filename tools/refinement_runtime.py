#!/usr/bin/env python3
"""Three counterbalanced uncached CPU runs; quality is not inferred from this video."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

from benchmark import environment
from refinement_study import ROOT, check_freeze, guard_file_writes, sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs/refinement')
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/refinement_protocol.json')
    args = parser.parse_args(argv)
    try:
        frozen = check_freeze(args.protocol, args.runs)
        recommendation = frozen['recommended_profile']['content']
        protocol = frozen['protocol']
        video, model = ROOT / 'test_video.mp4', ROOT / 'best.onnx'
        directory = args.runs / 'runtime'
        aggregate = directory / 'aggregate.json'
        guard_file_writes([aggregate, directory / 'measurements.json'], [video, model, args.protocol])
        if aggregate.exists():
            raise ValueError('Runtime aggregate already exists; use a new --runs folder for a new timing study')
        plans = {'baseline_v2': ('baseline_v2', {**protocol['shared_tracker'], 'association': 'two_stage', 'schedule': 'every', 'output_policy': 'active'}),
                 'quality': ('current', recommendation['args']), 'budget': ('current', recommendation['budget_args'])}
        records = []
        orders = [('baseline_v2', 'quality', 'budget'), ('quality', 'budget', 'baseline_v2'), ('budget', 'baseline_v2', 'quality')]

        def run(name, repetition, frames):
            executable, knobs = plans[name]
            output = directory / f'{name}_{repetition}'
            command = [str(ROOT / protocol['executables'][executable]), '--video', str(video), '--model', str(model),
                       '--frames', str(frames), '--threads', '8', '--output', str(output)]
            for key, value in knobs.items():
                command.extend(['--' + key.replace('_', '-'), str(int(value) if isinstance(value, bool) else value)])
            started = time.perf_counter()
            result = subprocess.run(command, env=environment(), capture_output=True, text=True)
            elapsed = time.perf_counter() - started
            if result.returncode:
                raise RuntimeError(result.stderr or result.stdout)
            summary = json.loads((output / 'summary.json').read_text())
            if summary['frames'] != frames or summary['mode'] == 'cached_detection_replay':
                raise ValueError('Timing requires complete live ONNX inference runs')
            return {'profile': name, 'repetition': repetition, 'wall_seconds': elapsed, 'wall_fps': frames / elapsed,
                    'detector_calls': summary['detector_calls'], 'frames': frames, 'summary': summary,
                    'command': command, 'executable_sha256': sha256(Path(command[0]))}

        # Explicit warmup excluded. Uncached refers to detector execution, not cold OS file caches.
        for name in plans:
            run(name, 'warmup_excluded', 16)
        for repetition, order in enumerate(orders, 1):
            for name in order:
                record = run(name, repetition, 240)
                records.append(record)
                print(f"Run {repetition} {name}: {record['wall_fps']:.3f} wall FPS, {record['detector_calls']} detector calls", flush=True)
        output = {'scope': 'Unlabelled original video, first240frames; uncached OpenCV DNN CPU inference,8threads. Excludes dataset quality evaluation; includes process/model initialization and disk logs, no GUI/rendering.16frame warmups excluded; OS caches may be warm.',
                  'counterbalanced_order': orders, 'video_sha256': sha256(video), 'model_sha256': sha256(model),
                  'measurements_file': 'measurements.json', 'profiles': {}}
        for name in plans:
            rows = [record for record in records if record['profile'] == name]
            fps = [record['wall_fps'] for record in rows]
            output['profiles'][name] = {'runs': len(rows), 'wall_fps_mean': statistics.mean(fps), 'wall_fps_min': min(fps),
                                        'wall_fps_max': max(fps), 'wall_fps_stdev': statistics.stdev(fps),
                                        'detector_calls_mean': statistics.mean(record['detector_calls'] for record in rows)}
        output['budget_to_baseline_mean_fps_ratio'] = output['profiles']['budget']['wall_fps_mean'] / output['profiles']['baseline_v2']['wall_fps_mean']
        output['quality_to_baseline_mean_fps_ratio'] = output['profiles']['quality']['wall_fps_mean'] / output['profiles']['baseline_v2']['wall_fps_mean']
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'measurements.json').write_text(json.dumps(records, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        aggregate.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f'error: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
