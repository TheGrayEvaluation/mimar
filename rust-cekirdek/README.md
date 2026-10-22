# Mimar — yapay zekânın SQLite'ı

Her programa gömülü, sıfır kurulumla çalışan, programa **yapay zekâ kazandıran**
açık kaynak çekirdek. Tek taşınabilir dosya (`.mimar`) = bir programın bütün AI'ı:
kararı, hafızası, becerisi. Kullanıldıkça öğrenir; tekrar eden soruları modele
sormadan **ücretsiz/anında** cevaplar.

- **Dilden bağımsız** (SQLite gibi): saf hashing, çeviri/eş-anlam/dil katmanı YOK.
- **Model bağımsız:** Ollama / LM Studio / API takılabilir; çekirdek model çağırmaz,
  **yönlendirir** (host çalıştırır).
- **Sıfır bağımlılık** çekirdek (Rust std) → sunucudan gömülüye her yere.

## Kurulum / derleme

```bash
cd rust-cekirdek
cargo build --release                 # çekirdek + CLI (mimar) — SIFIR bağımlılık
cargo build --release --features gui  # native GUI (mimar-gui) — eframe
```
Üretilen: `target/release/` → `mimar` (CLI), `mimar-gui` (GUI),
`mimar_core.dll/.so/.dylib` (C ABI), `mimar_core.lib` (statik), `include/mimar.h`.

CLI'yi sisteme kur: `cargo install --path .`

## CLI

```bash
mimar new    b.mimar --role "Kısa asistan." --guard "yatırım tavsiyesi"
mimar teach  b.mimar "iade süresi" "14 gün"
mimar ask    b.mimar "Ay'a uzaklık" --model llama3.2:3b   # yeni→LLM, tekrar→0ms
mimar resolve b.mimar "iade süresi"     # tipli direktif (model çağırmaz)
mimar serve  b.mimar --model llama3.2:3b   # MCP sunucusu (ajanlar mount eder)
mimar sign/verify/merge/consolidate/fit/policy ...
```

## GUI (yönetim konsolu)

```bash
mimar-gui b.mimar     # native pencere: gör/düzenle/sil/engelle, rol/guard/hafıza, Yeni/Aç
```

## Kütüphane olarak (programcılar)

**Rust:**
```rust
use mimar_core::store::Brain;
let mut b = Brain::open(Some("b.mimar"))?;
b.teach("soru", "cevap");
let d = b.resolve("soru");   // Directive: Fact/Rule/Tool/Stale/Miss
```

**C / Python / Node / Go (C ABI, `mimar_core.dll` + `include/mimar.h`):**
```python
import ctypes
m = ctypes.CDLL("mimar_core.dll")
m.mimar_open.restype = ctypes.c_void_p
h = m.mimar_open(b"b.mimar")
m.mimar_teach(h, b"soru", b"cevap")
# m.mimar_resolve(h, b"soru") -> direktif JSON (sonra mimar_free_string)
```

## Ajana montaj (MCP)

`.mcp.json` (Cursor / Antigravity / Windsurf):
```json
{ "mcpServers": { "mimar": {
  "command": "…/mimar.exe",
  "args": ["serve", "…/b.mimar", "--model", "llama3.2:3b"] } } }
```
Araçlar: `mimar_ask`, `mimar_teach`, `mimar_explain`, `mimar_feedback`.

## Felsefe

Mimar bir cache değil, bir **yönlendiricidir**: her soru için "neyi hatırlarım,
neye her seferinde bakarım, neyi üretirim, neyi yaparım" kararını verir
(Fact/Rule/Tool/Stale/Miss). Çekirdek kazanç: `kesin` katman — bedava, anında,
deterministik.

## Lisans

MIT — bkz. `LICENSE`.
