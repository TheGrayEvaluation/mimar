#!/usr/bin/env bash
# Mimar kurulum — derler ve `mimar` ikilisini PATH'e koyar (Linux/macOS).
set -euo pipefail
cd "$(dirname "$0")/rust-cekirdek"

echo "→ Mimar derleniyor (release, sıfır bağımlılık)…"
cargo build --release

BIN="target/release/mimar"
[ -f "$BIN" ] || { echo "HATA: $BIN yok"; exit 1; }

DEST="${PREFIX:-$HOME/.local/bin}"
mkdir -p "$DEST"
cp "$BIN" "$DEST/mimar"
chmod +x "$DEST/mimar"

echo "→ Kuruldu: $DEST/mimar"
case ":$PATH:" in
  *":$DEST:"*) echo "→ PATH'te, hazır: mimar new beyin.mimar" ;;
  *) echo "→ PATH'e ekleyin:  export PATH=\"$DEST:\$PATH\"" ;;
esac

# GUI isteğe bağlı:  ./kur.sh gui
if [ "${1:-}" = "gui" ]; then
  echo "→ GUI derleniyor (egui)…"
  cargo build --release --features gui --bin mimar-gui
  cp target/release/mimar-gui "$DEST/mimar-gui"
  echo "→ Kuruldu: $DEST/mimar-gui"
fi
