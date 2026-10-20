# UnbiasedStrata 2 build on Windows: the Makefile's targets with the same CMake flags, for PowerShell 5.1 or 7.
# Needs Visual Studio 2022 or newer with the C++ workload (its CMake and Ninja are used), a CUDA 12.x toolkit, and
# for `test` Git for Windows (bash, curl) and Python 3.
# Nothing is downloaded: ggml, llama, common and mtmd come from the third_party/llama.cpp submodule.
#
#   git clone <repo>; cd <repo>
#   git submodule update --init
#   .\make.ps1 build                 # build\bin\{strata,strata-poly,unbiased-strata}.exe, strata_kq.dll
#   .\make.ps1 test                  # the front end against a fake engine; no GPU needed
#   .\make.ps1 frontend              # the front end alone, without CUDA
#   .\make.ps1 clean
#
# Variables as in the Makefile, as parameters: -CudaArchs "89" builds for one card (several times faster);
# -Nvcc C:\path\nvcc.exe; -Jobs N; -Portable (AVX2 baseline and static MSVC runtime, to run on another PC);
# -BuildDir; -CmakeExtra "-DX=Y ...".
param(
    [Parameter(Position = 0)][ValidateSet('build', 'configure', 'frontend', 'test', 'clean')][string]$Target = 'build',
    [string]$BuildDir = 'build',
    # CUDA archs: GTX 10 (61), RTX 20 / GTX 16 (75), RTX 30 (86), RTX 40 (89). Any arch below 75 turns on upstream's
    # STRATA_EXPERIMENTAL_SM60. An arch the installed nvcc cannot compile is dropped with a note.
    [string]$CudaArchs = '61;75;86;89',
    [string]$Nvcc = '',
    [int]$Jobs = 0,
    [switch]$Portable,
    [string]$CmakeExtra = ''
)
# every native call is checked by its exit code; 'Stop' would turn a compiler's stderr into an error (PowerShell 5.1)
$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Targets = @('strata', 'strata-poly', 'strata_kq', 'unbiased-strata')

function Fail($msg) { Write-Host "make.ps1: $msg" -ForegroundColor Red; exit 1 }

function Run($exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { Fail "$exe failed (exit $LASTEXITCODE)" }
}

# ---- Visual Studio: its newest install (prerelease included), CMake and Ninja from it --------------------------
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) { Fail 'Visual Studio not found (no vswhere.exe); install VS 2022+ with "Desktop development with C++"' }
$vs = & $vswhere -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { Fail 'no Visual Studio with the C++ x64 tools' }
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
$vsCmake = Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake'

# nvcc: -Nvcc, else on PATH, else CUDA_PATH
if (-not $Nvcc) {
    $c = Get-Command nvcc.exe -ErrorAction SilentlyContinue
    if ($c) { $Nvcc = $c.Source } elseif ($env:CUDA_PATH -and (Test-Path "$env:CUDA_PATH\bin\nvcc.exe")) { $Nvcc = "$env:CUDA_PATH\bin\nvcc.exe" }
}
$needCuda = $Target -in @('build', 'configure')
if ($needCuda -and -not $Nvcc) { Fail 'nvcc not found on PATH or in CUDA_PATH; install a CUDA 12.x toolkit or pass -Nvcc C:\path\nvcc.exe' }

# Load a vcvars64 environment into this process. -vcvars_ver picks the MSVC toolset.
function Use-Toolset($ver) {
    $arg = if ($ver) { "-vcvars_ver=$ver" } else { '' }
    $lines = & cmd.exe /c "`"$vcvars`" $arg >nul 2>&1 && set"
    if ($LASTEXITCODE -ne 0) { return $false }
    foreach ($l in $lines) { if ($l -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] } }
    return $true
}

# One compiler for the C, C++ and CUDA host sides. nvcc accepts MSVC only up to a version (CUDA 12.x: the VS 2022
# 14.4x toolsets); the probe tries the installed toolsets newest first, the way the Makefile probes g++ versions.
$toolsets = @(Get-ChildItem (Join-Path $vs 'VC\Tools\MSVC') -Directory | Sort-Object { [version]$_.Name } -Descending | ForEach-Object Name)
$saved = @{}; Get-ChildItem env: | ForEach-Object { $saved[$_.Name] = $_.Value }
$chosen = $null
foreach ($t in $toolsets) {
    foreach ($k in @(Get-ChildItem env: | ForEach-Object Name)) { if (-not $saved.ContainsKey($k)) { Remove-Item "env:$k" } }
    foreach ($k in $saved.Keys) { Set-Item "env:$k" $saved[$k] }
    if (-not (Use-Toolset $t)) { continue }
    if (-not $needCuda) { $chosen = $t; break }
    $probe = Join-Path $env:TEMP "strata-hc-$PID.cu"
    Set-Content -Path $probe -Value 'int main(){return 0;}' -Encoding ASCII
    & $Nvcc -c $probe -o "$probe.obj" 2>&1 | Out-Null
    $ok = $LASTEXITCODE -eq 0
    Remove-Item $probe, "$probe.obj" -ErrorAction SilentlyContinue
    if ($ok) { $chosen = $t; break }
    Write-Host "make.ps1: nvcc does not accept MSVC $t; trying an older toolset"
}
if (-not $chosen) { Fail "no MSVC toolset that $Nvcc accepts (installed: $($toolsets -join ', ')); add the VS 2022 v143 build tools" }
$env:PATH = "$vsCmake\CMake\bin;$vsCmake\Ninja;$env:PATH"
Write-Host "make.ps1: MSVC $chosen$(if ($needCuda) { ", $Nvcc" })"

if ($Jobs -le 0) { $Jobs = [Math]::Min([int]$env:NUMBER_OF_PROCESSORS, 16) }   # capped as in the Makefile
$abs = Join-Path $Root $BuildDir
$common = @('-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DSTRATA_BUILD_TESTS=OFF', '-Wno-dev')
$extra = if ($CmakeExtra) { $CmakeExtra -split ' ' | Where-Object { $_ } } else { @() }

# A build folder copied or moved to another place keeps the old absolute paths in its CMake cache, and CMake refuses
# to configure it there. Its cache (not its built programs, which run from anywhere) is dropped and made again.
function Reset-MovedCache($dir) {
    $cache = Join-Path $Root "$dir\CMakeCache.txt"
    if (-not (Test-Path $cache)) { return }
    $line = Select-String -Path $cache -Pattern '^CMAKE_CACHEFILE_DIR:INTERNAL=(.*)$' | Select-Object -First 1
    if (-not $line) { return }
    $was = $line.Matches[0].Groups[1].Value -replace '/', '\'
    $now = (Join-Path $Root $dir) -replace '/', '\'
    if ($was.TrimEnd('\') -ne $now.TrimEnd('\')) {
        Write-Host "make.ps1: $dir was configured at $was; configuring it again here"
        Remove-Item $cache -Force
        Remove-Item (Join-Path $Root "$dir\CMakeFiles") -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Configure {
    if (-not (Test-Path 'third_party\llama.cpp\ggml\CMakeLists.txt')) { Fail "third_party/llama.cpp is empty: run 'git submodule update --init' first" }
    Reset-MovedCache $BuildDir
    $have = @(& $Nvcc --list-gpu-arch 2>$null | ForEach-Object { $_ -replace 'compute_', '' })
    $want = $CudaArchs -split ';' | Where-Object { $_ }
    if ($have.Count) {
        $dropped = @($want | Where-Object { $have -notcontains ($_ -split '-')[0] })
        if ($dropped.Count) { Write-Host "make.ps1: nvcc cannot compile CUDA arch(s) $($dropped -join ' '); building without them" }
        $want = @($want | Where-Object { $have -contains ($_ -split '-')[0] })
    }
    $archs = $want -join ';'
    $sm60 = @($want | Where-Object { [int](($_ -split '-')[0]) -lt 75 }).Count -gt 0
    $argv = @('-S', '.', '-B', $BuildDir) + $common + @(
        '-DSTRATA_ENABLE_CUDA=ON', '-DSTRATA_NATIVE_EXPERTS=ON', '-DSTRATA_MMQ_KQUANTS=ON', '-DSTRATA_Q6K_EXPERTS=ON',
        "-DCMAKE_CUDA_ARCHITECTURES=$archs", "-DCMAKE_CUDA_COMPILER=$($Nvcc -replace '\\','/')",
        # CMakeLists.txt #585: MSVC with a CUDA 12.x toolkit links every cudart symbol twice (LNK1169) with the default
        # static runtime beside CUDA::cudart; the shared runtime alone links
        '-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared',
        '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl',
        "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$($abs -replace '\\','/')/bin",
        "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$($abs -replace '\\','/')/bin")
    if ($sm60) { $argv += '-DSTRATA_EXPERIMENTAL_SM60=ON' }
    if ($Portable) { $argv += '-DSTRATA_PORTABLE=ON' }
    Run cmake ($argv + $extra)
}

# the CUDA runtime DLLs the engines load, copied beside them so build\bin runs from any shell
function Copy-CudaDlls {
    $bin = Join-Path (Split-Path -Parent $Nvcc) ''
    foreach ($p in @('cudart64_*.dll', 'cublas64_*.dll', 'cublasLt64_*.dll')) {
        Get-ChildItem $bin -Filter $p | ForEach-Object { Copy-Item $_.FullName (Join-Path $abs 'bin') -Force }
    }
}

switch ($Target) {
    'configure' { Configure }
    'build' {
        Configure
        Run cmake (@('--build', $BuildDir, '-j', "$Jobs", '--target') + $Targets)
        Copy-CudaDlls
        Write-Host 'built:'
        foreach ($f in @('strata.exe', 'strata-poly.exe', 'strata_kq.dll', 'unbiased-strata.exe')) { Write-Host "  $BuildDir\bin\$f" }
    }
    'frontend' {
        Reset-MovedCache "$BuildDir-frontend"
        # The front end alone, without CUDA: for a laptop or a CI box. Needs only a C++20 compiler and CMake.
        Run cmake (@('-S', '.', '-B', "$BuildDir-frontend") + $common + @('-DSTRATA_ENABLE_CUDA=OFF',
            "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$($abs -replace '\\','/')-frontend/bin", '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl') + $extra)
        Run cmake @('--build', "$BuildDir-frontend", '-j', "$Jobs", '--target', 'unbiased-strata')
    }
    'test' {
        Reset-MovedCache "$BuildDir-frontend"
        # The front end's tests, against a fake engine (frontend/tests/fake-engine.py), as `make test`.
        Run cmake (@('-S', '.', '-B', "$BuildDir-frontend") + $common + @('-DSTRATA_ENABLE_CUDA=OFF',
            "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$($abs -replace '\\','/')-frontend/bin", '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl') + $extra)
        Run cmake @('--build', "$BuildDir-frontend", '-j', "$Jobs", '--target', 'unbiased-strata', 'engine_test')
        Run "$abs-frontend\bin\engine_test.exe" @('frontend\tests\fake-engine.py')
        $bash = @("$env:ProgramFiles\Git\bin\bash.exe", "$env:ProgramFiles\Git\usr\bin\bash.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
        if (-not $bash) { Fail 'frontend/tests/e2e.sh needs bash: install Git for Windows' }
        Run $bash @('frontend/tests/e2e.sh', "$BuildDir-frontend/bin/unbiased-strata.exe")
    }
    'clean' { if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir } }
}
