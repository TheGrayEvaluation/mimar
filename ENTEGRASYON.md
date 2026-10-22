# Mimar — Entegrasyon Rehberi (bağlantı yolları)

Mimar'ı her yere nasıl bağlarsın. Hepsi aynı `.mimar` dosyası + aynı çekirdek.

## 1. Yerel model — Ollama / LM Studio
```bash
mimar ask b.mimar "soru" --model llama3.2:3b           # Ollama (127.0.0.1:11434)
mimar ask b.mimar "soru" --lmstudio --model <model>    # LM Studio (127.0.0.1:1234)
```

## 2. Bulut API — OpenRouter / OpenAI / Groq / Together (TLS)
Derleme: `cargo build --release --features cloud`
```bash
# OpenRouter (anahtar --key veya OPENROUTER_API_KEY)
mimar ask b.mimar "soru" --provider openrouter --key sk-or-... --model openai/gpt-4o-mini
# Herhangi bir OpenAI-uyumlu uç
mimar ask b.mimar "soru" --base-url https://api.openai.com/v1 --key sk-... --model gpt-4o-mini
```
Mantık: yeni soru buluta gider; **tekrarı 0 ms, beyinden, bulut çağrısı YOK.**

## 3. Kalıcı HTTP servisi (uzak / çok-kullanıcı / otomasyon)
```bash
mimar serve-http b.mimar --addr 0.0.0.0:8899 --token GIZLI_TOKEN
# okuma uçları serbest; YAZMA uçları "Authorization: Bearer GIZLI_TOKEN" ister
```
Uçlar:
| Yöntem | Uç | Yetki | Gövde (JSON) |
|---|---|---|---|
| GET | `/saglik` `/stats` | açık | — |
| POST | `/resolve` | açık | `{"query":"...","user":"ops."}` → direktif (Fact/Rule/Tool/Stale/Miss) |
| POST | `/redact` | açık | `{"text":"..."}` → PII maskeli |
| POST | `/teach` | **token** | `{"prompt":"...","answer":"..."}` |
| POST | `/define` | **token** | `{"query","answer","collection","tool","slots","ttl","scope"}` |
| POST | `/note` `/forget` `/policy` | **token** | ilgili alanlar |
| POST | `/batch` | **token** | `{"ops":[ {op:"teach",...}, ... ]}` (toplu) |

## 4. n8n / Zapier / Make / webhook
**HTTP Request** node'u ile:
- **Bilgi çek (ücretsiz):** `POST http://SUNUCU:8899/resolve`, gövde `{"query":"{{soru}}"}`.
  Dönen `tip` alanı: `fact` → `cevap`'ı kullan (model çağırma); `miss`/`stale` → kendi
  modelini/akışını çalıştır; `tool`/`rule` → ilgili node'u tetikle (slotlar JSON'da).
- **Öğret (token):** `POST .../teach` + Header `Authorization: Bearer GIZLI_TOKEN`,
  gövde `{"prompt":"{{soru}}","answer":"{{cevap}}"}`.
Böylece n8n akışında tekrar eden sorular Mimar'dan bedava gelir; yenisi öğrenilir.

## 5. Programcı — kütüphane (kendi uygulaman / API otomasyonun)
Mimar **yönlendirir**, modeli/araçları **sen** çağırırsın.
**C ABI** (`mimar_core.dll/.so` + `include/mimar.h`) → C/Python/Node/Go/C#:
```python
import ctypes, json
m = ctypes.CDLL("mimar_core.dll"); m.mimar_open.restype=ctypes.c_void_p
m.mimar_resolve.restype=ctypes.c_void_p; m.mimar_resolve.argtypes=[ctypes.c_void_p,ctypes.c_char_p]
h = m.mimar_open(b"b.mimar")
d = json.loads(ctypes.cast(m.mimar_resolve(h,"soru".encode()),ctypes.c_char_p).value)
if d["tip"]=="fact":  cevap = d["cevap"]          # bedava
else:                 cevap = benim_apim(soru)     # kendi modelin/API'in
                      m.mimar_teach(h,"soru".encode(),cevap.encode())  # öğret
```
**Rust:** `use mimar_core::store::Brain;` → `brain.resolve(q)` / `brain.teach(...)`.

## 6. Ajanlar — Cursor / Antigravity / Windsurf (MCP)
Proje köküne `.mcp.json`:
```json
{ "mcpServers": { "mimar": {
  "command": "…/mimar.exe", "args": ["serve","…/b.mimar","--model","llama3.2:3b"] } } }
```
+ ajan kural dosyası'ye "her soruda önce mimar_ask" kuralı → ajan otomatik Mimar'a danışır.

---
**Kurulum zorunlu mu?** Hayır. Kütüphane kullanınca Mimar programının parçası olur
(statik link = içine gömülü; dinamik = `.dll`'i yanında taşı). `.mimar` + programın
her yere taşınır; GUI gerekmez. Uzak yönetim için `serve-http` + token yeterli.
