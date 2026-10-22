# Paketleme

Mimar'ı dağıtmak için hazır paket tanımları. Yayın (release) yapıldıktan sonra
URL ve SHA256 alanları gerçek sürüm arşivleriyle doldurulur.

| Platform | Dosya | Komut |
|---|---|---|
| Rust / crates.io | `rust-cekirdek/Cargo.toml` | `cargo publish` |
| Debian / Ubuntu | `Cargo.toml` → `[package.metadata.deb]` | `cargo deb` (cargo-deb kurulu olmalı) |
| macOS / Linux (Homebrew) | `packaging/mimar.rb` | `brew install --build-from-source ./packaging/mimar.rb` |
| Windows (Scoop) | `packaging/scoop/mimar.json` | `scoop install ./packaging/scoop/mimar.json` |
| Python (pip) | `pyproject.toml` | `pip install .` |
| Node (npm) | `npm/package.json` | `npm install` (+ `koffi`) |
| Elle (Unix) | `kur.sh` | `./kur.sh` |
| Elle (Windows) | `kur.ps1` | `./kur.ps1` |

Binary release'ler GitHub Actions ile (`.github/workflows/ci.yml` → `release` işi)
`v*` etiketi atılınca 3 platform için otomatik üretilir.

> Not: Depo `github.com/TheGrayEvaluation/mimar`. Homebrew/Scoop manifestlerindeki
> `sha256`/`hash` alanları, bir sürüm (release) yayınlanınca o arşivin özetiyle doldurulur.
