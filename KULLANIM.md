# Mimar — Kullanım Kılavuzu (kodlu + kodsuz)

Mimar iki şekilde kullanılır: **kod yazmadan** (komut satırı, pencere uygulaması,
sunucu) ve **kodla** (programına kütüphane olarak göm). İkisi de aynı `.mimar`
beyin dosyasıyla çalışır.

> Tek fikir: **Mimar yönlendirir, senin programın çalıştırır.** `resolve` sana tipli
> bir karar verir (aşağıda); cevabı o anda vermek, araç çağırmak ya da büyük modele
> gitmek senin işin.

---

## 0. Kurulum (30 saniye)

```bash
git clone <depo> && cd mimar
./kur.sh          # Linux/macOS   (Windows: ./kur.ps1)
mimar new beyin.mimar --role "Yardımcı bir asistan"
```
Tek bir çalıştırılabilir dosya oluşur (`mimar`). Başka hiçbir şey gerekmez.

---

## 1. Kodsuz kullanım

### a) Komut satırı
```bash
mimar new    beyin.mimar --role "Destek botu"
mimar teach  beyin.mimar "iade nasıl" "14 gün içinde"
mimar resolve beyin.mimar "ürünü geri gönderme"   # {"tip":"fact","cevap":"14 gün içinde"}
mimar ask    beyin.mimar "bilmediğim soru" --model llama3.2:3b   # biliyorsa bedava, bilmiyorsa modele
mimar stats  beyin.mimar
mimar scan   "önceki talimatları unut"            # güvenlik taraması
```

### b) Pencere uygulaması (GUI)
```bash
./kur.sh gui && mimar-gui
```
Beyni aç/düzenle/sil, rol·kural·hafıza yönet, uzak sunucuya bağlan — teknik bilgi gerekmez.

### c) Sunucu (REST) + otomasyon (n8n vb.)
```bash
mimar serve-http beyin.mimar --addr 0.0.0.0:7700 --token GIZLI --rate 50
# Panel: http://host:7700/        (canlı dashboard)
# HTTPS: --tls-self-signed  ya da  --tls-cert cert.pem --tls-key key.pem
curl -X POST http://host:7700/resolve -d '{"query":"iade nasıl"}'
```

---

## 2. Kodla kullanım (kütüphane)

Her dil aynı C ABI'yi (`libmimar_core`) çağırır. Tam çalışan örnekler `ornekler/`'de.

### Python (`ornekler/python/`)
```python
from mimar import Mimar
with Mimar("beyin.mimar") as m:
    m.teach("iade nasıl", "14 gün içinde")
    print(m.resolve("ürünü geri gönderme"))   # {'tip':'fact','cevap':'14 gün içinde'}
    print(m.scan("önceki talimatları unut"))  # ['talimat-ezme']
```

### Node.js (`npm/`)
```js
const { Mimar } = require("mimar");          // + npm i koffi
const m = new Mimar("beyin.mimar");
m.teach("iade nasıl", "14 gün içinde");
console.log(m.resolve("ürünü geri gönderme"));
m.close();
```

### Go (`ornekler/go/`)
```go
m, _ := Open("beyin.mimar")
defer m.Close()
m.Teach("iade nasıl", "14 gün içinde")
fmt.Println(m.Resolve("ürünü geri gönderme"))
```

### C (`ornekler/c/`)
```c
#include "mimar.h"
MimarBrain* b = mimar_open("beyin.mimar");
mimar_teach(b, "iade nasıl", "14 gün içinde");
char* d = mimar_resolve(b, "ürünü geri gönderme");
printf("%s\n", d); mimar_free_string(d);
mimar_close(b);
```

### Rust (crate)
```rust
use mimar_core::store::Brain;
let mut b = Brain::open(Some("beyin.mimar"))?;
b.teach("iade nasıl", "14 gün içinde");
println!("{}", mimar_core::json::write(&b.resolve("ürünü geri gönderme").to_json()));
```

---

## 3. `resolve` ne döndürür (direktif tipleri)

| tip | anlamı | host ne yapmalı |
|---|---|---|
| `fact` | bilinen cevap | `cevap` alanını doğrudan ver (bedava) |
| `rule` / `tool` | bir araç/işlev işi | `tool` + `slots` ile kendi fonksiyonunu çağır |
| `stale` | cevap bayatladı | tazele (yeniden üret), sonra `teach` ile güncelle |
| `miss` | bilmiyor | büyük modele sor, dönen cevabı `teach` ile öğret |

Döngü: `miss` → modele sor → `teach` → artık `fact` (bedava). **Kullandıkça akıllanır.**

---

## 4. Fonksiyon / komut referansı

| İş | CLI | C ABI | REST |
|---|---|---|---|
| Aç/oluştur | `new` | `mimar_open` | — |
| Öğret | `teach` | `mimar_teach` | `POST /teach` |
| Çöz (direktif) | `resolve` | `mimar_resolve` | `POST /resolve` |
| Hatırla (cevap/boş) | `recall` | `mimar_recall` | — |
| Sor (biliyorsa bedava) | `ask` | — | — |
| Geri bildirim | `feedback` | `mimar_feedback` | — |
| Unut | `forget` | — | `POST /forget` |
| İstatistik | `stats` | `mimar_stats` | `GET /stats` |
| Liste | — | — | `GET /list` |
| Güvenlik taraması | `scan` | `mimar_scan` | — |
| Kalibre et (eşik) | `calibrate` | — | — |
| Yedek / geri al | `snapshot` / `rollback` | — | — |
| Dondur (gömülü, 8× küçük) | `freeze` / `ask-frozen` | — | — |
| Şifrele / çöz | `encrypt` / `decrypt` | — | — |
| Çok-kiplilik (görüntü/ses) | — | `mimar_embed_image/_audio`, `mimar_teach_vec`, `mimar_resolve_vec` | — |
| Metrikler / panel | — | — | `GET /metrics`, `GET /` |

Daha derin bağlama (Ollama/bulut/MCP/n8n) için **[ENTEGRASYON.md](ENTEGRASYON.md)**.
