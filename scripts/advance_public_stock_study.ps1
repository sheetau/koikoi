param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("v2", "v3")]
    [string]$Version,
    [Parameter(Mandatory = $true)]
    [int]$ScreenProcessId
)

# Advance only the already registered screen, with its unchanged confirmation.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$studyPython = Join-Path (Get-Location) ".venv/Scripts/python.exe"
if ($Version -eq "v2") {
    $studyPath = "runs/public_stock_residual_v2"
    $runnerModule = "scripts.run_public_stock_residual"
    $exportModule = "scripts.export_public_stock_results"
    $confirmModule = "scripts.confirm_public_stock_residual"
    $confirmPath = "runs/public_stock_residual_confirmation_v2"
} else {
    $studyPath = "runs/public_stock_rounds_v3"
    $runnerModule = "scripts.run_public_stock_rounds"
    $exportModule = "scripts.export_public_stock_rounds"
    $confirmModule = "scripts.confirm_public_stock_rounds"
    $confirmPath = "runs/public_stock_rounds_confirmation_v3"
}
$screenEvidence = "$studyPath/screen_evidence.json"
$finalEvidence = "docs/paper/generated/public_stock_${Version}_evidence.json"
if (Test-Path -LiteralPath $confirmPath) {
    throw "Confirmation already exists; inspect it instead of restarting."
}
$screenProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $ScreenProcessId"
if ($null -ne $screenProcess) {
    if ($screenProcess.Name -ne "python.exe" -or
        -not $screenProcess.CommandLine.Contains($runnerModule) -or
        -not $screenProcess.CommandLine.Contains($studyPath)) {
        throw "The supplied PID does not identify the registered screen."
    }
    Write-Output "Waiting for $Version screen process $ScreenProcessId."
    $waitingProcess = Get-Process -Id $ScreenProcessId -ErrorAction SilentlyContinue
    if ($null -ne $waitingProcess) { $waitingProcess | Wait-Process }
}
if (-not (Test-Path -LiteralPath "$studyPath/screen_result.json")) {
    throw "Screen stopped without a complete result; no confirmation launched."
}
& $studyPython -X utf8 -m $exportModule --study $studyPath --out $screenEvidence
if ($LASTEXITCODE -ne 0) { throw "Independent screen reduction failed." }
$screenSummary = Get-Content -LiteralPath $screenEvidence -Encoding UTF8 -Raw | ConvertFrom-Json
if ($screenSummary.confirmation_gate_passed -eq $true) {
    Write-Output "Verified $Version gate passed; starting its reserved confirmation."
    & $studyPython -X utf8 -m $confirmModule --study $studyPath --out $confirmPath
    if ($LASTEXITCODE -ne 0) { throw "Confirmation stopped; inspect before any recovery." }
    & $studyPython -X utf8 -m $exportModule --study $studyPath --confirmation $confirmPath --out $finalEvidence
} else {
    Write-Output "Verified $Version gate failed; reserved confirmation remains unused."
    & $studyPython -X utf8 -m $exportModule --study $studyPath --out $finalEvidence
}
if ($LASTEXITCODE -ne 0) { throw "Final reduction failed." }
Write-Output "Completed $Version evidence: $finalEvidence"
