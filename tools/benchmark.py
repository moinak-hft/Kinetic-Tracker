#!/usr/bin/env python3
"""Run reproducible cached-detection experiments; never label replay as live FPS."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from evaluate import evaluate_sequence, box_iou, guard_output_paths, read_mot

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def environment():
    env=os.environ.copy()
    for candidate in [env.get('KINETIC_TOOLCHAIN',''),r'D:\Tools\msys64\ucrt64',r'C:\msys64\ucrt64']:
        if candidate and (Path(candidate)/'bin').exists():
            env['PATH']=str(Path(candidate)/'bin')+os.pathsep+env.get('PATH','');break
    env['OPENBLAS_NUM_THREADS']='1'
    return env

def discovery(gt_path,det_path,frame_log,frames,time_path):
    gt,_=read_mot(Path(gt_path),frames,True)
    selected={int(r['frame']) for r in csv.DictReader(Path(frame_log).open()) if int(r['detector_ran'])}
    times={int(r['frame']):float(r['timestamp']) for r in csv.DictReader(Path(time_path).open())}
    det={}
    for r in csv.DictReader(Path(det_path).open()):
        if float(r['score'])>=.08:
            det.setdefault(int(r['frame']),[]).append([float(r[k]) for k in ('x','y','w','h')])
    first,last,found={},{},{}
    for n in range(1,frames+1):
        rows=gt.get(n,[])
        for ident,_ in rows:first.setdefault(ident,n);last[ident]=n
        if n not in selected or not rows or not det.get(n):continue
        similarity=box_iou([r[1] for r in rows],det[n])
        score=(similarity>=.5)*1000+similarity
        a,b=linear_sum_assignment(-score)
        for i,j in zip(a,b):
            if similarity[i,j]>=.5:found.setdefault(rows[i][0],n)
    delays=[1000*(times[found[i]]-times[first[i]]) for i in first if i in found]
    frame_delays=[found[i]-first[i] for i in first if i in found]
    short=[i for i in first if times[last[i]]-times[first[i]]<=.4]
    return {'definition':'first selected detector box matched one-to-one to each GT identity at IoU>=0.5; detector score>=0.08; separate from tracking identity quality',
            'targets':len(first),'missed_targets':sum(i not in found for i in first),
            'short_targets':len(short),'missed_short_targets':sum(i not in found for i in short),
            'median_discovery_ms':float(np.median(delays)) if delays else None,
            'p95_discovery_ms':float(np.percentile(delays,95)) if delays else None,
            'median_discovery_frames':float(np.median(frame_delays)) if frame_delays else None,
            'p95_discovery_frames':float(np.percentile(frame_delays,95)) if frame_delays else None,
            'timing_note':'Frame delays are primary for real sources with unverified nominal capture timing.'}

def run_one(sequence,config,output,source='provided',force=False):
    sequence=Path(sequence);meta=json.loads((sequence/'metadata.json').read_text())
    output=Path(output)
    if source=='onnx':
        det=ROOT/'runs'/'detector_cache'/meta['name']/'detections.csv'
    else:
        det=sequence/meta.get('detections',meta.get('oracle_detections','detections.csv'))
    if not det.exists():raise FileNotFoundError(det)
    guard_output_paths([det,sequence/meta['ground_truth'],sequence/meta['timestamp_file'],sequence/'metadata.json'],
                       [output/name for name in ('mot.txt','telemetry.csv','frames.csv','detections.csv',
                                                 'summary.json','metrics.json','experiment.json')])
    output.mkdir(parents=True,exist_ok=True)
    command=[str(ROOT/'build'/('kinetic_tracker.exe' if os.name=='nt' else 'kinetic_tracker')),
             '--detections',str(det),
             '--frame-times',str(sequence/meta['timestamp_file']),'--frames',str(meta['num_frames']),
             '--fps',str(meta['fps']),'--output',str(output),
             '--min-hits','1','--iou','.1','--prediction-age','.24','--max-age','.75',
             '--high','.35','--low','.08','--output-policy','all']
    if config.get('schedule')=='adaptive':command.extend(['--images',str(sequence/meta['images'])])
    for key,value in config.items():
        if key!='name':command.extend(['--'+key.replace('_','-'),str(value)])
    record_path=output/'experiment.json'
    fingerprint={'command':command,'detector_sha256':sha(det),'gt_sha256':sha(sequence/meta['ground_truth']),
                 'frame_times_sha256':sha(sequence/meta['timestamp_file']),
                 'executable_sha256':sha(command[0]),'evaluator_sha256':sha(Path(__file__).with_name('evaluate.py')),
                 'config':config,'source':source,'sequence_metadata':meta}
    if not force and record_path.exists():
        old=json.loads(record_path.read_text())
        if all(old.get(k)==fingerprint[k] for k in fingerprint):return old
    started=time.time()
    proc=subprocess.run(command,env=environment(),capture_output=True,text=True)
    if proc.returncode:raise RuntimeError(f'{meta["name"]}: {proc.stderr}')
    metrics=evaluate_sequence(sequence/meta['ground_truth'],output/'mot.txt',meta['num_frames'])
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    summary=json.loads((output/'summary.json').read_text())
    fingerprint.update({'metrics':metrics,'summary':summary,
                        'discovery':discovery(sequence/meta['ground_truth'],det,output/'frames.csv',meta['num_frames'],sequence/meta['timestamp_file']),
                        'elapsed_seconds_including_evaluation':time.time()-started})
    record_path.write_text(json.dumps(fingerprint,indent=2,allow_nan=False)+'\n')
    return fingerprint

def aggregate(records):
    out={}
    for name in sorted({r['config']['name'] for r in records}):
        group=[r for r in records if r['config']['name']==name]
        frames=sum(r['summary']['frames'] for r in group);calls=sum(r['summary']['detector_calls'] for r in group)
        out[name]={'sequences':len(group),'frames':frames,'detector_calls':calls,'detector_rate':calls/frames,
                   **{k:float(np.mean([r['metrics'][k] for r in group])) for k in ['HOTA','AssA','DetA','IDF1','MOTA']},
                   **{k:sum(r['metrics'][k] for r in group) for k in ['IDSW','Frag','FP','FN']},
                   **{k:sum(r['discovery'][k] for r in group) for k in ['targets','missed_targets','short_targets','missed_short_targets']},
                   'aggregation':'HOTA/IDF1 are unweighted sequence means, NOT pooled official benchmark scores; counts summed; detector rate frame-weighted'}
    return out


def matched_uniform_config(adaptive_config, detector_calls):
    """Keep tracker/output settings unchanged for a detector-count control."""
    matched = dict(adaptive_config)
    matched.update(name='matched_'+adaptive_config['name'], schedule='uniform',
                   uniform_calls=detector_calls)
    return matched


DEFAULTS=[{'name':'every_sort','schedule':'every','association':'sort'},
          {'name':'every_two_stage','schedule':'every','association':'two_stage'},
          {'name':'fixed2','schedule':'fixed','interval':2,'association':'two_stage'},
          {'name':'fixed4','schedule':'fixed','interval':4,'association':'two_stage'},
          {'name':'fixed8','schedule':'fixed','interval':8,'association':'two_stage'},
          {'name':'covariance','schedule':'covariance','risk_threshold':.18,'max_gap':8,'association':'two_stage'},
          {'name':'risk_only','schedule':'risk','risk_threshold':.18,'max_gap':8,'association':'two_stage'},
          {'name':'adaptive','schedule':'adaptive','risk_threshold':.18,'novelty_threshold':.00008,'max_gap':8,'association':'two_stage'}]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['synthetic','native_bee','gmot40'],default='synthetic')
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--source',choices=['provided','onnx'],default='provided')
    p.add_argument('--configs',type=Path,help='JSON array of named scheduler/association argument dictionaries')
    p.add_argument('--output',type=Path,default=ROOT/'runs'/'benchmark')
    p.add_argument('--force',action='store_true')
    p.add_argument('--matched',action='store_true',help='Add content-independent uniform schedules with the exact per-sequence adaptive call counts')
    args=p.parse_args();splits=json.loads((ROOT/'configs'/'splits.json').read_text())
    guard_output_paths([ROOT/'configs'/'splits.json',args.configs],
                       [args.output/'aggregate.json',args.output/'aggregate.csv'])
    configs=json.loads(args.configs.read_text()) if args.configs else DEFAULTS
    records=[]
    for item in splits[args.dataset][args.split]:
        sequence=ROOT/item
        for config in configs:
            record=run_one(sequence,config,args.output/sequence.name/config['name'],args.source,args.force)
            records.append(record)
            print(f'{sequence.name:35} {config["name"]:20} HOTA={record["metrics"]["HOTA"]:.3f} IDF1={record["metrics"]["IDF1"]:.3f} calls={record["summary"]["detector_rate"]:.3f}',flush=True)
            if args.matched and config.get('schedule')=='adaptive':
                matched=matched_uniform_config(config,record['summary']['detector_calls'])
                uniform=run_one(sequence,matched,args.output/sequence.name/matched['name'],args.source,args.force)
                records.append(uniform)
                print(f'{sequence.name:35} {matched["name"]:20} HOTA={uniform["metrics"]["HOTA"]:.3f} IDF1={uniform["metrics"]["IDF1"]:.3f} calls={uniform["summary"]["detector_rate"]:.3f}',flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    report={'dataset':args.dataset,'split':args.split,'source':args.source,'configurations':configs,'aggregate':aggregate(records),
            'experiments':[str(Path(r['command'][r['command'].index('--output')+1])/'experiment.json') for r in records]}
    (args.output/'aggregate.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    fields=['name','sequences','frames','detector_rate','HOTA','IDF1','AssA','IDSW','FP','FN','missed_targets','missed_short_targets']
    with (args.output/'aggregate.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for name,r in report['aggregate'].items():w.writerow({k:name if k=='name' else r[k] for k in fields})
    print(json.dumps(report['aggregate'],indent=2))

if __name__=='__main__':main()
