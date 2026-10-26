$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$probe = [System.Net.Sockets.TcpClient]::new()
try {
    $probe.Connect('127.0.0.1', 8000)
    $portInUse = $true
} catch {
    $portInUse = $false
} finally {
    $probe.Dispose()
}
if ($portInUse) {
    try {
        $existing = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/status' -TimeoutSec 3
    } catch {
        throw 'Port 8000 is already in use by another service.'
    }
    if ($existing.ready -and $existing.model.backend -eq 'llama.cpp CUDA') {
        Write-Output 'The llama.cpp server is already running at http://127.0.0.1:8000.'
        return
    }
    throw 'Port 8000 is already in use by another server. Stop it before starting llama.cpp.'
}
$env:MODEL_BACKEND = 'llama_cpp'
& "$PSScriptRoot\start.ps1"
