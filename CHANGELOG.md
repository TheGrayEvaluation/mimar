# Değişiklik Günlüğü

Biçim: [Keep a Changelog] · Sürümleme: [SemVer]

## [0.1.0] — 2026-10 (deneysel/erken)

İlk genel sürüm. Yapay zekânın SQLite'ı: her programa gömülü, modelden bağımsız,
sıfır-bağımlılık AI çekirdeği.

### Eklendi
- **Çekirdek**: `.mimar` tek-dosya beyin biçimi (özgün ikili), planlayıcı/yönlendirici
  (Fact/Rule/Tool/Stale/Miss direktifleri), sıfır-bağımlılık Rust motoru.
- **Güvenli eşleşme**: birebir-anahtar + konformal kalibrasyon (`calibrate`, tavan 0.85)
  + marj-çekimser + geri bildirim ("benzerlik ≠ aynı cevap").
- **Embedder**: varsayılan hashing (dilden bağımsız) + opt-in nöral (Ollama bge-m3).
- **Arama**: HNSW indeks (yüksek recall, ölçekte hızlı).
- **Davranış damıtma**: `capture`/`fidelity` — bir modeli yakala, başka modelde üret, ölç.
- **Çok-kiplilik**: veri/görüntü/ses parmak izi (`modality`, FFI `embed_image/_audio`,
  `teach_vec`/`resolve_vec`).
- **Gömülü**: int8 nicemleme + `QuantPack` (`freeze`/`ask-frozen`), `--no-default-features`
  ağsız çekirdek.
- **Sunucu**: `serve-http` (REST), RBAC (okuma/yazma token), rate-limit, canlı dashboard,
  `/metrics`, çok-instance (yazıcı+replika), WAL dayanıklılık.
- **Sunucu TLS** (`tls` özelliği): rustls, kendinden-imzalı veya PEM.
- **Güvenlik**: at-rest şifreleme (`encrypt`/`decrypt`), prompt-injection taraması
  (`scan`), guard, imza (HMAC), PII maskeleme.
- **Yedek**: `snapshot` / `rollback` (point-in-time).
- **Arayüzler**: CLI, native GUI (egui, uzak panel), MCP sunucusu, C ABI (+ Python/Node/
  Go/C örnekleri).
- **Paketleme**: crates.io hazır, cargo-deb, Homebrew, Scoop, pip, npm; CI (3 OS + wasm +
  tls + binary release).

### Bilinen sınırlar (dürüst)
- Hash embedder parafrazda zayıf; nöral embedder Ollama servisi ister (0ms/offline değil).
- Kripto AES-GCM değil (SHA-256 CTR+HMAC); injection taraması sezgisel.
- wasm32/MCU ve Go-cgo yalnız uygun araç zincirinde derlenir (CI'da wasm işi var).
- Gerçek LoRA eğitimi iskelet (derin YZ dışarıda — tasarım).
- Henüz dış/bağımsız doğrulama ve üretim pilotu yapılmadı.
