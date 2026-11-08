#!/usr/bin/env python3
"""Build and test without changing Windows PowerShell execution policy."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=ROOT/'build')
    parser.add_argument('--toolchain-root', type=Path)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    env = os.environ.copy()
    configure = ['cmake', '-S', str(ROOT), '-B', str(args.build_dir), '-DCMAKE_BUILD_TYPE=Release']
    if os.name == 'nt':
        candidates = [args.toolchain_root, env.get('KINETIC_TOOLCHAIN'),
                      r'D:\Tools\msys64\ucrt64', r'C:\msys64\ucrt64']
        toolchain = next((Path(p) for p in candidates if p and (Path(p)/'lib/cmake/opencv4/OpenCVConfig.cmake').is_file()), None)
        if toolchain is None:
            parser.error('Pass --toolchain-root pointing at an existing MSYS2 UCRT64/OpenCV installation.')
        env['PATH'] = str(toolchain/'bin')+os.pathsep+env.get('PATH','')
        configure.extend(['-G','Ninja',f'-DCMAKE_CXX_COMPILER={(toolchain/"bin/g++.exe").as_posix()}',
                          f'-DOpenCV_DIR={(toolchain/"lib/cmake/opencv4").as_posix()}'])
    python = ROOT/('.venv/Scripts/python.exe' if os.name == 'nt' else '.venv/bin/python')
    configure.append(f'-DPython3_EXECUTABLE={(python if python.exists() else Path(sys.executable)).as_posix()}')
    for command in [configure, ['cmake','--build',str(args.build_dir),'--parallel',str(args.jobs)],
                    ['ctest','--test-dir',str(args.build_dir),'--output-on-failure']]:
        subprocess.run(command,env=env,check=True,cwd=ROOT)
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode)
