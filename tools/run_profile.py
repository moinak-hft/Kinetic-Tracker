#!/usr/bin/env python3
"""Run the validation-selected quality or compute-budget profile on an ONNX video."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
from benchmark import ROOT, environment

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,default=ROOT/'configs/recommended_profile.json')
    p.add_argument('--budget',action='store_true',help='Use the validation-selected lower-call option instead of every-frame quality mode')
    p.add_argument('--video',type=Path,default=ROOT/'test_video.mp4')
    p.add_argument('--model',type=Path,default=ROOT/'best.onnx')
    p.add_argument('--output',type=Path)
    p.add_argument('--output-video',type=Path)
    p.add_argument('--frames',type=int)
    p.add_argument('--preview',action='store_true')
    a=p.parse_args();profile=json.loads(a.profile.read_text(encoding='utf-8'))
    knobs=profile['budget_args'] if a.budget else profile['args']
    output=a.output or ROOT/'runs'/('demo_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    executable=ROOT/'build'/('kinetic_tracker.exe' if os.name=='nt' else 'kinetic_tracker')
    command=[str(executable),'--video',str(a.video),'--model',str(a.model),'--output',str(output),
             '--threads',str(min(8,os.cpu_count() or 1))]
    for key,value in knobs.items():
        if isinstance(value,bool):value=int(value)
        command.extend(['--'+key.replace('_','-'),str(value)])
    if a.frames is not None:command.extend(['--frames',str(a.frames)])
    if a.output_video:command.extend(['--output-video',str(a.output_video)])
    if a.preview:command.append('--preview')
    subprocess.run(command,env=environment(),check=True)
    print('Saved:',output)
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except subprocess.CalledProcessError as error:raise SystemExit(error.returncode)
