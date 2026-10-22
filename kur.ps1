# Mimar kurulum — derler ve `mimar.exe`'yi kullanıcı bin klasörüne koyar (Windows).
$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\rust-cekirdek"

Write-Host "-> Mimar derleniyor (release, sifir bagimlilik)..."
cargo build --release
if ($LASTEXITCODE -ne 0) { throw "derleme basarisiz" }

$bin = "target\release\mimar.exe"
if (-not (Test-Path $bin)) { throw "HATA: $bin yok" }

$dest = Join-Path $env:LOCALAPPDATA "Mimar\bin"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Copy-Item $bin (Join-Path $dest "mimar.exe") -Force
Write-Host "-> Kuruldu: $dest\mimar.exe"

if ($args -contains "gui") {
  Write-Host "-> GUI derleniyor (egui)..."
  cargo build --release --features gui --bin mimar-gui
  Copy-Item "target\release\mimar-gui.exe" (Join-Path $dest "mimar-gui.exe") -Force
  Write-Host "-> Kuruldu: $dest\mimar-gui.exe"
}

if ($env:PATH -notlike "*$dest*") {
  Write-Host "-> PATH'e eklemek icin (kalici):"
  Write-Host "   [Environment]::SetEnvironmentVariable('PATH', `"$dest;`$env:PATH`", 'User')"
}
