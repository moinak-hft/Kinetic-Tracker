#!/usr/bin/env python3
"""Rebuild the merged v2 baseline from pinned Git source, without touching main."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BASE = 'e2d2343fd3814d1f4fe114172373d801ed7ca7f8'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'work/kinetic_baseline_v2.exe' if os.name=='nt' else ROOT/'work/kinetic_baseline_v2')
    a=p.parse_args();source=ROOT/'work/reproducible_v2_source';build=ROOT/'work/reproducible_v2_build'
    blob=subprocess.check_output(['git','archive',BASE,'CMakeLists.txt','src','include'],cwd=ROOT)
    source.mkdir(parents=True,exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        for item in archive.getmembers():
            dest=(source/item.name).resolve()
            if not dest.is_relative_to(source.resolve()) or not (item.isfile() or item.isdir()):
                raise ValueError('Unsafe archived Git source member')
        archive.extractall(source,filter='data')
    env=os.environ.copy();cmd=['cmake','-S',str(source),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release','-DBUILD_TESTING=OFF']
    if os.name=='nt':
        candidates=[env.get('KINETIC_TOOLCHAIN'),r'D:\Tools\msys64\ucrt64',r'C:\msys64\ucrt64']
        toolchain=next((Path(v) for v in candidates if v and (Path(v)/'lib/cmake/opencv4/OpenCVConfig.cmake').exists()),None)
        if not toolchain:raise ValueError('Existing OpenCV toolchain unavailable')
        env['PATH']=str(toolchain/'bin')+os.pathsep+env.get('PATH','')
        cmd.extend(['-G','Ninja',f'-DCMAKE_CXX_COMPILER={(toolchain/"bin/g++.exe").as_posix()}',f'-DOpenCV_DIR={(toolchain/"lib/cmake/opencv4").as_posix()}'])
    subprocess.run(cmd,env=env,check=True)
    subprocess.run(['cmake','--build',str(build),'--target','kinetic_tracker','--parallel','2'],env=env,check=True)
    binary=build/('kinetic_tracker.exe' if os.name=='nt' else 'kinetic_tracker')
    # Output is restricted to the project's ignored work directory.
    if not a.output.resolve().is_relative_to((ROOT/'work').resolve()):raise ValueError('Baseline output must stay inside project work/')
    a.output.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(binary,a.output)
    manifest={'source_commit':BASE,'git_source_archive_sha256':hashlib.sha256(blob).hexdigest(),
              'executable_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'configure_command':cmd,
              'note':'Pinned merged v2 source; same compiler/OpenCV backend. PE build timestamps may change binary hashes without changing numeric tracking outputs.'}
    a.output.with_suffix('.provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(a.output)

if __name__=='__main__':main()
