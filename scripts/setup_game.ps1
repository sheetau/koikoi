$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$engineDir = Join-Path $projectRoot 'external/godot'
$engineZip = Join-Path $engineDir 'godot.zip'
$engineExe = Join-Path $engineDir 'Godot_v4.7.2-stable_win64_console.exe'
$releaseUrl = 'https://github.com/godotengine/godot-builds/releases/download/4.7.2-stable/Godot_v4.7.2-stable_win64.exe.zip'
$expectedHash = '83DECD58FDF67B9D657958A1AE6BF1929C20785315A81EFFE245874CDC57ACB709BF868E00778A96984338C1B29DAFDB453C6847747694621C6ECF5DA2259993'

New-Item -ItemType Directory -Force -Path $engineDir | Out-Null
if (-not (Test-Path -LiteralPath $engineZip)) {
    Write-Host 'Downloading pinned Godot 4.7.2...'
    Invoke-WebRequest -Uri $releaseUrl -OutFile $engineZip
}
if ((Get-FileHash -LiteralPath $engineZip -Algorithm SHA512).Hash -ne $expectedHash) {
    throw 'Godot archive checksum mismatch. The archive was not executed.'
}
Expand-Archive -LiteralPath $engineZip -DestinationPath $engineDir -Force
& $engineExe --headless --path (Join-Path $projectRoot 'game') --editor --import --quit
if ($LASTEXITCODE -ne 0) { throw 'Godot asset import failed.' }
Write-Host 'Ready. Double-click play.cmd to play.'
