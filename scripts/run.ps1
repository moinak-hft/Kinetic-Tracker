$ErrorActionPreference = 'Stop'
$trackerRoot = Split-Path -Parent $PSScriptRoot
foreach ($candidate in @($env:KINETIC_TOOLCHAIN, 'D:\Tools\msys64\ucrt64', 'C:\msys64\ucrt64')) {
    if ($candidate -and (Test-Path -LiteralPath "$candidate\lib\cmake\opencv4\OpenCVConfig.cmake")) { $env:PATH = "$candidate\bin;$env:PATH"; break }
}
& "$trackerRoot\build\kinetic_tracker.exe" @args
exit $LASTEXITCODE
