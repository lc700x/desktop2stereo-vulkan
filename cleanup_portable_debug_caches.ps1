[CmdletBinding()]
param(
    [switch]$Delete
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path.TrimEnd([char]'\')
$repoPrefix = $repoRoot + '\'
$Candidates = @{}

$srcPath = Join-Path $repoRoot 'src'
$pythonRoots = @()
if (Test-Path -LiteralPath $srcPath) {
    $pythonRoots = @(
        Get-ChildItem -LiteralPath $srcPath -Directory -Filter 'python*' -Force `
            -ErrorAction SilentlyContinue |
        ForEach-Object { (Resolve-Path -LiteralPath $_.FullName).Path }
    )
}

function Add-CacheCandidate {
    param(
        [string]$Path,
        [string]$Reason
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    $resolvedPath = (Resolve-Path -LiteralPath $Path).Path

    if (-not $resolvedPath.StartsWith(
        $repoPrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "拒绝处理仓库外路径：$resolvedPath"
    }

    foreach ($pythonRoot in $pythonRoots) {
        $pythonPrefix = $pythonRoot.TrimEnd([char]'\') + '\'
        if (
            $resolvedPath.Equals(
                $pythonRoot,
                [System.StringComparison]::OrdinalIgnoreCase
            ) -or
            $resolvedPath.StartsWith(
                $pythonPrefix,
                [System.StringComparison]::OrdinalIgnoreCase
            )
        ) {
            throw "拒绝处理 Python 环境：$resolvedPath"
        }
    }

    $item = Get-Item -LiteralPath $resolvedPath -Force
    if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "拒绝处理符号链接或联接点：$resolvedPath"
    }

    $Candidates[$resolvedPath] = $Reason
}

$fixedCaches = @(
    @{ Path = '.pytest_cache'; Reason = 'pytest 缓存' },
    @{ Path = '.serena\cache'; Reason = 'Serena 缓存' },
    @{ Path = '__pycache__'; Reason = 'Python 字节码缓存' },
    @{ Path = 'outputs\rocm_compare\metavr_cache'; Reason = 'MetaVR npm 缓存' },
    @{ Path = 'outputs\rocm_compare\current\models'; Reason = 'ROCm 对比副本模型缓存' },
    @{ Path = 'outputs\rocm_compare\current\jit_cache'; Reason = 'ROCm 对比副本 JIT 缓存' },
    @{ Path = 'outputs\rocm_compare\old\models'; Reason = '旧版对比副本模型缓存' },
    @{ Path = 'outputs\rocm_compare\old\jit_cache'; Reason = '旧版对比副本 JIT 缓存' }
)

foreach ($cache in $fixedCaches) {
    $path = Join-Path $repoRoot $cache.Path
    Add-CacheCandidate -Path $path -Reason $cache.Reason
}

$pythonCacheRoots = @(
    'src\desktop2stereo',
    'src\tools',
    'tests',
    'scripts',
    '.tmp',
    'outputs\rocm_compare'
)

foreach ($relativeRoot in $pythonCacheRoots) {
    $searchRoot = Join-Path $repoRoot $relativeRoot
    if (Test-Path -LiteralPath $searchRoot) {
        Get-ChildItem -LiteralPath $searchRoot -Directory -Filter '__pycache__' `
            -Recurse -Force -ErrorAction SilentlyContinue |
            ForEach-Object {
                Add-CacheCandidate -Path $_.FullName -Reason 'Python 字节码缓存'
            }
    }
}

$portableRoot = Join-Path $repoRoot 'artifacts\windows-portable'
if (Test-Path -LiteralPath $portableRoot) {
    Get-ChildItem -LiteralPath $portableRoot -Directory -Filter 'portable-*' -Force `
        -ErrorAction SilentlyContinue |
        ForEach-Object {
            Add-CacheCandidate -Path $_.FullName -Reason '便携版构建输出'
        }
}

$rows = @()
foreach ($path in $Candidates.Keys) {
    $files = @(Get-ChildItem -LiteralPath $path -File -Force -Recurse `
        -ErrorAction SilentlyContinue)
    $bytes = ($files | Measure-Object -Property Length -Sum).Sum
    if ($null -eq $bytes) {
        $bytes = 0
    }

    $rows += [pscustomobject]@{
        Path   = $path
        Reason = $Candidates[$path]
        Files  = $files.Count
        MiB    = [math]::Round(($bytes / 1MB), 1)
    }
}

if ($rows.Count -eq 0) {
    Write-Host '没有找到匹配的便携版构建输出或调试缓存。'
    return
}

$rows | Sort-Object Path | Format-Table -AutoSize

if (-not $Delete) {
    Write-Host "`n这是预览，没有删除文件。确认清单后运行：.\cleanup_portable_debug_caches.ps1 -Delete"
    return
}

$confirmation = Read-Host '确认删除以上缓存请输入 DELETE'
if ($confirmation -cne 'DELETE') {
    Write-Host '已取消，没有删除文件。'
    return
}

foreach ($path in @($Candidates.Keys)) {
    Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction Stop
    Write-Host "已删除：$path"
}

Write-Host "`n清理完成。Python 环境未纳入删除范围。"