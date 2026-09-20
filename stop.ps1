$ErrorActionPreference = 'Stop'
$labRoot = $PSScriptRoot
foreach ($name in @('frontend', 'backend')) {
    $recordPath = Join-Path $labRoot ".runtime\$name.json"
    if (!(Test-Path -LiteralPath $recordPath)) { continue }
    $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
    if ($record.root -ne $labRoot) { throw 'Process record belongs to another workspace.' }
    $process = Get-Process -Id $record.pid -ErrorAction SilentlyContinue
    if (!$process) { continue }
    if ([Math]::Abs(($process.StartTime.ToUniversalTime() - ([DateTime]$record.started).ToUniversalTime()).TotalSeconds) -gt 1) {
        Write-Host "$name PID was reused; leaving that process running."
        continue
    }
    # Record the descendants before stopping this verified launcher's process.
    $processSnapshot = Get-CimInstance Win32_Process
    $owned = [System.Collections.Generic.List[int]]::new()
    $owned.Add([int]$record.pid)
    for ($i=0; $i -lt $owned.Count; $i++) {
        foreach ($child in ($processSnapshot | Where-Object { $_.ParentProcessId -eq $owned[$i] })) { $owned.Add([int]$child.ProcessId) }
    }
    for ($i=$owned.Count-1; $i -ge 0; $i--) { Stop-Process -Id $owned[$i] -ErrorAction SilentlyContinue }
    Remove-Item -LiteralPath $recordPath
    Write-Host "$name stopped."
}
