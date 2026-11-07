#!/usr/bin/env python3
"""Measure uncached CPU execution on the original supplied video, counterbalanced."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
from benchmark import ROOT,environment,sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames',type=int,default=240);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--threads',type=int,default=8);p.add_argument('--output',type=Path,default=ROOT/'runs'/'runtime')
    a=p.parse_args();selection=json.loads((ROOT/'configs'/'scheduler_selection.json').read_text())['selected'][1]
    executable=ROOT/'build'/('kinetic_tracker.exe' if os.name=='nt' else 'kinetic_tracker')
    records=[]
    for repeat in range(a.repeats):
        names=['every','adaptive50'] if repeat%2==0 else ['adaptive50','every']
        for name in names:
            out=a.output/f'{name}_{repeat}'
            cmd=[str(executable),'--video',str(ROOT/'test_video.mp4'),'--model',str(ROOT/'best.onnx'),
                 '--frames',str(a.frames),'--threads',str(a.threads),'--association','two_stage',
                 '--output-policy','active','--min-hits','1','--iou','.1','--prediction-age','.24',
                 '--output',str(out),'--schedule','every' if name=='every' else 'adaptive']
            if name!='every':
                for key in ('target_rate','risk_threshold','novelty_threshold','max_gap'):cmd.extend(['--'+key.replace('_','-'),str(selection[key])])
            subprocess.run(cmd,env=environment(),check=True)
            record=json.loads((out/'summary.json').read_text());record.update({'name':name,'repeat':repeat,'command':cmd})
            records.append(record)
    aggregate={}
    for name in ('every','adaptive50'):
        group=[r for r in records if r['name']==name]
        aggregate[name]={'runs':len(group),'frames_per_run':group[0]['frames'],
                         'mean_wall_seconds':statistics.mean(r['wall_seconds'] for r in group),
                         'sd_wall_seconds':statistics.stdev(r['wall_seconds'] for r in group) if len(group)>1 else 0,
                         'mean_processing_fps':statistics.mean(r['processing_fps'] for r in group),
                         'mean_detector_rate':statistics.mean(r['detector_rate'] for r in group),
                         'median_frame_ms_across_runs':statistics.median(r['median_frame_ms'] for r in group),
                         'median_p95_frame_ms_across_runs':statistics.median(r['p95_frame_ms'] for r in group)}
    report={'input':'original unlabelled10second1280x72024FPS sample; no tracking-accuracy inference',
            'method':'counterbalanced sequential runs, no preview or video encoding; CPU OpenCV backend; initialization and one warmup inference separate',
            'source_sha256':sha(ROOT/'test_video.mp4'),'model_sha256':sha(ROOT/'best.onnx'),'executable_sha256':sha(executable),
            'threads':a.threads,'aggregate':aggregate,'runs':records,
            'mean_wall_speedup':aggregate['every']['mean_wall_seconds']/aggregate['adaptive50']['mean_wall_seconds'],
            'limitations':['One short input and one laptop; no GPU, energy or field-deployment result.',
                           'Fewer detector calls trade accuracy for compute; this unlabelled input cannot establish identity quality.',
                           'p95 values are per-frame compute latency, not camera-to-display latency or a hard realtime guarantee.']}
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'aggregate.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'aggregate':aggregate,'speedup':report['mean_wall_speedup']},indent=2))

if __name__=='__main__':main()
