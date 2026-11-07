param([string]$ToolchainRoot = '', [string]$BuildDir = 'build')
$ErrorActionPreference = 'Stop'
$trackerRoot = Split-Path -Parent $PSScriptRoot
if (-not $ToolchainRoot) {
    foreach ($candidate in @($env:KINETIC_TOOLCHAIN, 'D:\Tools\msys64\ucrt64', 'C:\msys64\ucrt64')) {
        if ($candidate -and (Test-Path -LiteralPath "$candidate\lib\cmake\opencv4\OpenCVConfig.cmake")) { $ToolchainRoot = $candidate; break }
    }
}
if (-not $ToolchainRoot) { throw 'Pass -ToolchainRoot with your MSYS2 UCRT64 directory containing OpenCV.' }
$env:PATH = "$ToolchainRoot\bin;$env:PATH"
$configure = @('-S', $trackerRoot, '-B', "$trackerRoot\$BuildDir", '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', "-DCMAKE_CXX_COMPILER=$ToolchainRoot/bin/g++.exe", "-DOpenCV_DIR=$ToolchainRoot/lib/cmake/opencv4")
if (Test-Path -LiteralPath "$trackerRoot\.venv\Scripts\python.exe") { $configure += "-DPython3_EXECUTABLE=$trackerRoot/.venv/Scripts/python.exe" }
& cmake @configure
if ($LASTEXITCODE) { throw 'CMake configuration failed.' }
& cmake --build "$trackerRoot\$BuildDir" --parallel 2
if ($LASTEXITCODE) { throw 'Build failed.' }
& ctest --test-dir "$trackerRoot\$BuildDir" --output-on-failure
if ($LASTEXITCODE) { throw 'Tests failed.' }
