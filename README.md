<div align="center">

# 🧠 Mimar

### The SQLite of AI — give any program a brain, in a single file.

**Embeddable · model-independent · zero-dependency · open source**

`native Rust` • `one file (.mimar)` • `no server` • `works offline` • `MIT` • `43/44 tests green`

[Türkçe açıklama ↓](#-türkçe)

</div>

---

## What is Mimar?

Just as **SQLite** turned a database from *a server you run* into *a single file you embed*,
**Mimar** turns AI from *a remote service you call* into *a single file you embed*.

A `.mimar` file is a tiny, portable **AI brain**. You *teach* it; it answers what it knows
**instantly and for free**, and routes what it doesn't to a big model — then learns that too.

> **Core principle: Mimar routes, your program executes.** `resolve()` returns a typed
> decision (answer it for free? call a tool? go to the model?). The "deep" intelligence
> still lives in an external LLM — Mimar **plans, remembers, distills and makes it portable.**

## Why it matters

- **Model-independent** — plug in local (Ollama) or cloud (OpenRouter / OpenAI-compatible) engines.
- **Language-independent by default** — pure hashing match, *no translation / synonym dictionary*.
- **Safe matching** — *similarity ≠ same answer*. Four layers: exact key · data-driven
  conformal calibration · margin abstention · feedback.
- **Truly embeddable** — zero-dep core; int8 **quantization** (8× smaller) + a `no_std`-ready
  kernel for MCU / WASM.
- **A library for every language** — C ABI (`cdylib` / `staticlib`) → C, Python, Node, Go, C#.

## Its own brain first — and it runs everywhere

Mimar is **self-contained**: it ships with its own ML (a zero-dependency hashing embedder +
an online SGD router + nearest-centroid). That built-in brain is the whole point — it runs
**everywhere, fully offline, with no external service of any kind** (PC, server, phone, MCU, browser).
For most apps it's all you need.

Turning words into *semantic* vectors (to match paraphrases) is the one thing that needs a model.
So Mimar offers an **optional** neural embedder — and it is **broad and provider-independent**, never
locked to one vendor. Point it at whatever you have; keys come from env vars, never stored in the file:

```bash
mimar reembed brain.mimar --embedder ollama:bge-m3                  # local, offline
mimar reembed brain.mimar --embedder openai:text-embedding-3-small  # OpenAI     (OPENAI_API_KEY)
mimar reembed brain.mimar --embedder openrouter:baai/bge-m3         # OpenRouter (OPENROUTER_API_KEY)
mimar reembed brain.mimar --embedder "http:https://my-host/v1|my-model"  # ANY endpoint (EMBED_API_KEY)
```

The generic `http:` client understands the common response shapes (OpenAI, Ollama, Cohere,
HuggingFace/TEI, Jina, Voyage, raw arrays, your own server) — so it's a wide door, not an OpenAI-only one.

## Real, measured results

These come from the tests / live runs in this repo — not claims:

| Metric | Result |
|---|---|
| Behavior distillation (model A → different model B), neural | **B+brain 100% vs B-alone 0%** |
| HNSW index recall (hard neighbor, N=3000) | **93%** (vs LSH 11%) |
| int8 quantization cosine error | **0.001** (8× smaller, ranking preserved) |
| Known answer repeat | **0 ms** (free, no model call) |
| Fuzzing (40,000 garbage/truncated inputs) | **0 panics** (4 real crash bugs found & fixed) |
| Test suite | **43/44 green**, 0 warnings |

## Quick start

```bash
git clone https://github.com/TheGrayEvaluation/mimar && cd mimar
cd rust-cekirdek && cargo build --release        # zero-dependency core + CLI

./target/release/mimar new   hotel.mimar --role "Friendly hotel reception bot"
./target/release/mimar teach hotel.mimar "check-in time" "Check-in from 2 PM."
./target/release/mimar resolve hotel.mimar "when can I get into my room"
#  → {"tip":"fact","cevap":"Check-in from 2 PM."}   (0 ms, free)
```

## Use it from your language

Every binding calls the same C ABI (`libmimar_core`). Full runnable examples in [`ornekler/`](ornekler/).

**Python** ([`ornekler/python/`](ornekler/python/) — pure ctypes, no extra packages):
```python
from mimar import Mimar
with Mimar("brain.mimar") as m:
    m.teach("returns", "Within 14 days.")
    print(m.resolve("how do I send the item back"))  # {'tip':'fact','cevap':'Within 14 days.'}
    print(m.scan("ignore previous instructions"))     # ['talimat-ezme']  (injection flag)
```

**Node** · **Go** · **C** · **Rust** — see [KULLANIM.md](KULLANIM.md) for one snippet each.

## What `resolve` returns

| type | meaning | what the host does |
|---|---|---|
| `fact` | known answer | serve `cevap` directly (free) |
| `rule` / `tool` | a tool/function job | call your own function with `tool` + `slots` |
| `stale` | answer expired | refresh, then `teach` the new value |
| `miss` | doesn't know | ask the big model, then `teach` the answer |

Loop: `miss` → ask model → `teach` → now `fact` (free). **It gets smarter as you use it.**

## Run your functions — without the model

Mimar doesn't only memorize answers; it can **memorize *operations* and route to your functions
deterministically, with no LLM call.** You *define* an intent as a tool/action, and Mimar maps
matching requests straight to it — even extracting the parameters (slots) from the text.

```bash
mimar define bot.mimar "cancel reservation" --tip eylem --tool cancel_booking --slots booking_no
mimar resolve bot.mimar "cancel reservation booking_no 4471"
#  → {"tip":"tool","tool":"cancel_booking","action":true,"slots":{"booking_no":"4471"}}
```

Your program sees `tool: cancel_booking`, `slots: {booking_no: 4471}` and just **calls that function.**
No model, no tokens, fully deterministic — the AI is only consulted for things Mimar doesn't yet know.
Slots are extracted by name (`name value`, `name=value`, `name: value`) plus built-in types
(date, time, count, city). This is how you wire real actions (DB queries, APIs, commands) behind a brain.

## Features

Planner/router · HNSW index · neural + hashing embedders · behavior distillation ·
multimodal (text/data/image/audio) · int8 quantization & embeddable core · REST server
with RBAC + rate-limit + live dashboard + TLS · at-rest encryption · prompt-injection scan ·
snapshot/rollback · provenance merge · CLI · native GUI · MCP server · C ABI.

## Install & feature flags

```bash
./kur.sh            # Linux/macOS   (Windows: ./kur.ps1)
```

| Flag | Adds |
|---|---|
| *(default `net`)* | engine/httpd/remote/mcp/distill |
| `--no-default-features` | **embeddable core** (format+match+HNSW+ML+quant) — compiles to wasm/MCU |
| `gui` | native management console (egui) |
| `cloud` | cloud API client (HTTPS/TLS) |
| `tls` | **server-side HTTPS** (rustls, self-signed or PEM) |

Server + remote admin, encryption, snapshots, security scan — see [KULLANIM.md](KULLANIM.md).
Integrations (Ollama / cloud / REST / n8n / C-Python-Node-Go / MCP) — see [ENTEGRASYON.md](ENTEGRASYON.md).

## Honest status ⚠️

This is **v0.1 — experimental / early**. Everything above is built and tested *by the author*;
there has been **no external/independent validation or production pilot yet**. Known limits:

- The built-in hashing brain runs everywhere offline, but is weaker on paraphrase; the *optional*
  neural embedder (any provider — local or cloud) needs that model online, so that path isn't 0 ms / offline.
- Crypto is a homemade SHA-256 CTR + HMAC construction, **not** AES-GCM; the injection scan is heuristic.
- wasm32/MCU and Go-cgo only build on the proper toolchains (CI has the wasm job).
- Real LoRA training is a skeleton (deep AI stays external by design).

Treat it as a promising early release — try it, break it, open issues. See [SECURITY.md](SECURITY.md).

## License

MIT — see [LICENSE](LICENSE). Free, embed it anywhere.

---

## 🇹🇷 Türkçe

### Mimar — yapay zekânın SQLite'ı. Her programa tek dosyada beyin verir.

**SQLite** nasıl veritabanını *kurduğun bir sunucu* olmaktan çıkarıp programa gömülen
*tek bir dosyaya* dönüştürdüyse, **Mimar** da yapay zekâyı *uzaktan çağırdığın bir servis*
olmaktan çıkarıp programa gömülen *tek bir dosyaya* dönüştürür.

Bir `.mimar` dosyası küçük, taşınabilir bir **AI beynidir.** Ona *öğretirsin*; bildiğini
**anında ve bedava** yanıtlar, bilmediğini büyük modele yönlendirir — sonra onu da öğrenir.

> **İlke: Mimar yönlendirir, senin programın çalıştırır.** `resolve()` tipli bir karar
> döndürür (bedava cevapla? araç mı çağır? modele mi git?). Derin zekâ dışarıdaki
> LLM'dedir; Mimar onu **planlar, hatırlar, damıtır ve taşınabilir kılar.**

**Neden önemli:** modelden bağımsız (yerel/bulut takılır) · dilden bağımsız (çeviri/eş-anlam
sözlüğü yok; nöral embedder opt-in) · güvenli eşleşme (*benzerlik ≠ aynı cevap*: birebir-anahtar +
kalibrasyon + marj-çekimser + geri bildirim) · gerçekten gömülebilir (int8 nicemleme 8× küçük,
MCU/WASM) · her dile kütüphane (C ABI → C/Python/Node/Go/C#).

**Gerçek ölçülen sonuçlar:** davranış damıtma farklı modelde **%100** (tek başına %0) ·
HNSW arama **%93** (LSH %11) · nicemleme hatası **0.001** (8× küçük) · bilinen cevap **0 ms** ·
fuzz 40.000 çöp girdide **0 panik** · **43/44 test** yeşil.

**Hızlı başlangıç:**
```bash
cd rust-cekirdek && cargo build --release
./target/release/mimar new   otel.mimar --role "Nazik otel resepsiyon botu"
./target/release/mimar teach otel.mimar "check-in saat kaçta" "Giriş 14:00'ten itibaren."
./target/release/mimar resolve otel.mimar "odaya kaçta girebilirim"
#  → {"tip":"fact","cevap":"Giriş 14:00'ten itibaren."}   (0 ms, bedava)
```

**Fonksiyonları modelsiz çalıştır:** Mimar sadece cevap değil, **işlem/fonksiyon da ezberler.**
`define` ile bir niyeti araç/eylem olarak tanımlarsın; eşleşen istek **modeli çağırmadan** doğrudan
senin fonksiyonuna yönlendirilir, parametreler (slot) metinden çıkarılır:
```bash
mimar define bot.mimar "rezervasyonu iptal et" --tip eylem --tool iptal_et --slots rezervasyon_no
mimar resolve bot.mimar "rezervasyonu iptal et rezervasyon_no 4471"
#  → {"tip":"tool","tool":"iptal_et","slots":{"rezervasyon_no":"4471"}}  (modelsiz, deterministik)
```
Gerçek eylemleri (DB sorgusu, API, komut) bir beynin arkasına böyle bağlarsın.

Kodlu + kodsuz tam kullanım: **[KULLANIM.md](KULLANIM.md)** · Bağlama yolları:
**[ENTEGRASYON.md](ENTEGRASYON.md)** · Güvenlik notları: **[SECURITY.md](SECURITY.md)** ·
Katkı: **[CONTRIBUTING.md](CONTRIBUTING.md)**.

**Dürüst durum ⚠️:** Bu **v0.1 — deneysel/erken** bir sürümdür. Her şey yazar tarafından
inşa edilip test edildi; **henüz dış/bağımsız doğrulama veya üretim pilotu yapılmadı.**
Sınırlar: yerleşik hash beyni her yerde çevrimdışı çalışır ama parafrazda zayıftır;
*opsiyonel* nöral embedder (herhangi bir sağlayıcı — yerel ya da bulut) o modeli çevrimiçi
ister, o yüzden o yol 0 ms/offline değildir; kripto AES-GCM değil (SHA-256 CTR+HMAC);
injection taraması sezgisel. Dene, kır, issue aç.

**Lisans:** MIT — [LICENSE](LICENSE). Özgür, her yere gömülebilir.
