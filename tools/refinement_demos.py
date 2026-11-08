#!/usr/bin/env python3
"""Render all additional category comparisons without selecting by performance."""
import argparse
import hashlib
import json
from pathlib import Path

from make_demo import make_demo
from refinement_study import ROOT, case_key, check_freeze


def render_coverage(cases, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    figure, axes = plt.subplots(3, 3, figsize=(12, 9.8))
    figure.patch.set_facecolor('#f6f8fb')
    sources = []
    for axis, case in zip(axes.flat, cases):
        directory = ROOT / case['path']
        metadata = json.loads((directory / 'metadata.json').read_text())
        image = directory / 'img1/000001.jpg'
        with Image.open(image) as pixels:
            axis.imshow(pixels)
        axis.set_title(f"{metadata['name']} · {metadata['num_frames']} frames", fontsize=11, pad=9)
        axis.axis('off')
        sources.append({'sequence': metadata['name'], 'frames': metadata['num_frames'],
                        'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest()})
    figure.suptitle('Nine non-insect categories · frozen confirmation set', x=.04, y=.96,
                   ha='left', fontsize=17, fontweight='bold')
    figure.text(.04, .025, 'GMOT-40 / Bai et al., CVPR2021 · Original JPEGs · CC BY-NC-SA4.0\n'
                'Author-provided frozen detections; no new neural training. One clip/category is not a full benchmark.',
                fontsize=9, color='#475569')
    figure.subplots_adjust(left=.035, right=.985, top=.89, bottom=.10, hspace=.35, wspace=.06)
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / 'coverage.png', dpi=150)
    plt.close(figure)
    (output / 'coverage.attribution.json').write_text(json.dumps({'license': 'CC BY-NC-SA4.0',
        'source': 'https://spritea.github.io/GMOT40/download.html',
        'attribution': 'Hexin Bai, Wensheng Cheng, Peng Chu, Kai Zhang and Haibin Ling. GMOT-40, CVPR2021.',
        'images': sources}, indent=2) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs/refinement')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/refinement')
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/refinement_protocol.json')
    parser.add_argument('--group', choices=['confirmation', 'exploratory'], default='confirmation')
    args = parser.parse_args(argv)
    try:
        frozen = check_freeze(args.protocol, args.runs)
        profile = frozen['recommended_profile']['content']['budget_profile']
        jobs = []
        if args.group == 'confirmation':
            for case in frozen['protocol']['confirmation_cases']:
                directory = args.runs / 'confirmation' / case_key(case)
                name = Path(case['path']).name
                jobs.append((ROOT / case['path'], directory / 'baseline_v2/fixed2/active', directory / profile / 'fixed2/active',
                             args.output / f'demo_{name}.mp4', 'V2: detect every2frames', 'Hybrid motion: detect every2frames'))
        else:
            directory = args.runs / 'exploratory/gmot40__insect-3'
            jobs.append((ROOT / 'data/gmot40/insect-3', directory / 'baseline_v2/fixed2/active',
                         directory / profile / 'fixed2/active', args.output / 'demo_insect.mp4',
                         'V2: detect every2frames', 'Hybrid motion: detect every2frames'))
            directory = args.runs / 'exploratory/synthetic__test__background_motion' / profile
            jobs.append((ROOT / 'data/synthetic/test/background_motion', directory / 'adaptive50_legacy_active/active',
                         directory / 'adaptive50_robust_active/active', args.output / 'demo_background.mp4',
                         'Legacy scene cue, active risk', 'Robust scene cue, active risk'))
        for sequence, left, right, output, left_label, right_label in jobs:
            result = make_demo(sequence, left, right, output, left_label, right_label, 640)
            print(f"Rendered {output.name}: {result['frames']}frames, all source frames retained", flush=True)
        if args.group == 'confirmation':
            render_coverage(frozen['protocol']['confirmation_cases'], args.output)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f'error: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
