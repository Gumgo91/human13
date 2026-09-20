param([switch]$OpenBrowser)
$ErrorActionPreference = 'Stop'
$labRoot = $PSScriptRoot
$labPython = Join-Path $labRoot '.venv\Scripts\python.exe'
$labWeb = Join-Path $labRoot 'web'
$labCli = Join-Path $labWeb 'node_modules\vinext\dist\cli.js'
$labNode = (Get-Command node -ErrorAction Stop).Source
$labRuntime = Join-Path $labRoot '.runtime'

if (!(Test-Path -LiteralPath $labPython) -or !(Test-Path -LiteralPath $labCli)) {
    throw 'Dependencies are missing. Follow README.md setup instructions first.'
}
New-Item -ItemType Directory -Path $labRuntime -Force | Out-Null

function Test-LabEndpoint([string]$Url, [string]$Marker) {
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
        return $response.StatusCode -eq 200 -and $response.Content.Contains($Marker)
    } catch { return $false }
}

function Start-LabProcess([string]$Name, [string]$Executable, [string]$Arguments, [string]$Directory) {
    $process = Start-Process -FilePath $Executable -ArgumentList $Arguments -WorkingDirectory $Directory -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $labRuntime "$Name.out.log") -RedirectStandardError (Join-Path $labRuntime "$Name.err.log")
    @{pid=$process.Id; started=$process.StartTime.ToUniversalTime().ToString('o'); root=$labRoot} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $labRuntime "$Name.json") -Encoding UTF8
    return $process
}

if (!(Test-LabEndpoint 'http://127.0.0.1:8000/api/health' 'local_only')) {
    $labBackend = Start-LabProcess 'backend' $labPython '-m uvicorn backend.app:app --host 127.0.0.1 --port 8000' $labRoot
}
if (!(Test-LabEndpoint 'http://localhost:5173/' 'human13')) {
    $labFrontend = Start-LabProcess 'frontend' $labNode ('"' + $labCli + '" dev --host 127.0.0.1 --port 5173') $labWeb
}

for ($attempt=0; $attempt -lt 30; $attempt++) {
    if ((Test-LabEndpoint 'http://127.0.0.1:8000/api/health' 'local_only') -and (Test-LabEndpoint 'http://localhost:5173/' 'human13')) {
        Write-Host 'Human13 is ready: http://localhost:5173/'
        Write-Host 'Local API: http://127.0.0.1:8000/docs'
        if ($OpenBrowser) { Start-Process 'http://localhost:5173/' }
        exit 0
    }
    if (($labBackend -and $labBackend.HasExited) -or ($labFrontend -and $labFrontend.HasExited)) {
        throw 'A server exited. See .runtime/*.err.log; a port may already be in use.'
    }
    Start-Sleep -Seconds 1
}
throw 'Startup timed out. See .runtime/*.err.log.'
