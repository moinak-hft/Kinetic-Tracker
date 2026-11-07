#!/usr/bin/env python3
"""Cache real ONNX detections from a frozen split; inference time remains explicit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from benchmark import ROOT,environment,sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='native_bee')
    p.add_argument('--split',choices=['validation','test'],required=True)
    p.add_argument('--model',type=Path,default=ROOT/'best.onnx')
    p.add_argument('--threads',type=int,default=2)
    args=p.parse_args();splits=json.loads((ROOT/'configs'/'splits.json').read_text())
    executable=ROOT/'build'/('kinetic_tracker.exe' if os.name=='nt' else 'kinetic_tracker')
    for item in splits[args.dataset][args.split]:
        seq=ROOT/item;meta=json.loads((seq/'metadata.json').read_text());out=ROOT/'runs'/'detector_cache'/meta['name']
        manifest=out/'cache_manifest.json'
        identity={'model_sha256':sha(args.model),'images_sha256':meta['image_content_sha256'],
                  'frame_times_sha256':sha(seq/meta['timestamp_file']),
                  'executable_sha256':sha(executable),'threads':args.threads}
        if manifest.exists() and json.loads(manifest.read_text()).get('identity')==identity:
            print('reuse',meta['name'],flush=True);continue
        command=[str(executable),'--images',str(seq/meta['images']),
                 '--frame-times',str(seq/meta['timestamp_file']),'--fps',str(meta['fps']),
                 '--frames',str(meta['num_frames']),'--model',str(args.model),'--output',str(out),
                 '--schedule','every','--min-hits','1','--threads',str(args.threads)]
        print('infer',meta['name'],meta['num_frames'],'frames',flush=True)
        subprocess.run(command,check=True,env=environment())
        manifest.write_text(json.dumps({'identity':identity,'command':command,'split':args.split,
                                        'dataset':args.dataset,'note':'Original pest model; training dataset provenance unavailable. '+('Native-bee images are processed ROIs, not raw scene footage.' if args.dataset=='native_bee' else 'Natural GMOT-40 images; capture timing is nominal, not verified.')},indent=2)+'\n')

if __name__=='__main__':main()
