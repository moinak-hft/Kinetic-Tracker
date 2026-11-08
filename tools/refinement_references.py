#!/usr/bin/env python3
"""Run predeclared official tracking references on the frozen confirmation set."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

from benchmark import environment
from evaluate import evaluate_sequence
from refinement_study import ROOT, check_freeze, case_inputs, guard_file_writes, sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs/refinement')
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/refinement_protocol.json')
    args = parser.parse_args(argv)
    try:
        frozen = check_freeze(args.protocol, args.runs)
        settings = frozen['protocol']['reference_controls']
        records = []
        for case in frozen['protocol']['confirmation_cases']:
            sequence, meta, detections, gt, times, inputs = case_inputs(case)
            for algorithm in ('ocsort', 'bytetrack'):
                output = args.runs / 'reference_confirmation' / meta['name'] / algorithm
                mot, manifest, metrics = [output / name for name in ('mot.txt', 'manifest.json', 'metrics.json')]
                guard_file_writes([mot, manifest, metrics], [detections, gt, times, sequence / 'metadata.json', args.protocol])
                command = [sys.executable, str(ROOT / 'tools/run_reference.py'), '--algorithm', algorithm,
                           '--detections', str(detections), '--output', str(mot), '--manifest', str(manifest),
                           '--frames', str(meta['num_frames']), '--frame-times', str(times), '--fps', str(meta['fps'])]
                for key, value in settings[algorithm].items():
                    if isinstance(value, bool):
                        if value:
                            command.append('--' + key.replace('_', '-'))
                    else:
                        command.extend(['--' + key.replace('_', '-'), str(value)])
                result = subprocess.run(command, env=environment(), capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError(result.stderr or result.stdout)
                scores = evaluate_sequence(gt, mot, meta['num_frames'])
                metrics.write_text(json.dumps(scores, indent=2, allow_nan=False) + '\n', encoding='utf-8')
                records.append({'sequence': meta['name'], 'algorithm': algorithm, 'metrics': scores,
                                'inputs': inputs, 'manifest_sha256': sha256(manifest), 'mot_sha256': sha256(mot)})
                print(f"{meta['name']} {algorithm}: HOTA={100*scores['HOTA']:.2f}, IDF1={100*scores['IDF1']:.2f}", flush=True)
        aggregate = args.runs / 'reference_confirmation/aggregate.json'
        guard_file_writes([aggregate], [args.protocol])
        aggregate.write_text(json.dumps({'protocol_sha256': frozen['protocol_sha256'], 'settings': settings,
                                         'experiments': records}, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f'error: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
