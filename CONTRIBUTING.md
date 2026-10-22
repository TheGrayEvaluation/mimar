# Katkı Rehberi

Teşekkürler! Mimar sıfır-bağımlılık, gömülebilir bir AI çekirdeğidir; katkılar bu
özü korumalıdır.

## Geliştirme
```bash
cd rust-cekirdek
cargo test --lib                 # tüm testler (varsayılan)
cargo build --release            # çekirdek + CLI
cargo build --no-default-features --lib   # GÖMÜLEBİLİR çekirdek (net yok) — kırılmamalı
cargo test --features tls --lib  # sunucu TLS yolu
```

## Değişmez kurallar (özü koru)
1. **Çekirdek sıfır bağımlılık.** `net`/`gui`/`cloud`/`tls` dışındaki her şey yalnız
   `std`. Yeni bir dış crate ancak opsiyonel bir özellik arkasında eklenir.
2. **Dilden bağımsızlık.** Varsayılan eşleşme hashing'dir; çeviri/eş-anlam sözlüğü
   EKLENMEZ. Nöral embedder opt-in kalır.
3. **Mimar yönlendirir, host çalıştırır.** Çekirdek/FFI modeli KENDİSİ çağırmaz.
4. **İddia değil kanıt.** Her davranış bir testle gösterilir; "çalışıyor" demeden önce
   `cargo test` yeşil olmalı. Rastgele/çöp girdiye karşı panik yok (bkz. fuzz testi).
5. **Güvenlik katmanı korunur.** Birebir-anahtar + kalibrasyon + marj-çekimser +
   geri bildirim; bunları zayıflatan değişiklik testle gerekçelendirilmelidir.

## PR beklentileri
- Yeni davranış → yeni test. Hata düzeltmesi → düzeltmeyi kilitleyen regresyon testi.
- `cargo build` 0 uyarı. Tüm özellik kombinasyonları derlenir.
- Kullanıcıya görünen değişiklik → `CHANGELOG.md` güncellenir.

## Mimarı
`rust-cekirdek/` ana motordur. Diğer diller (`ornekler/`, `npm/`) ince binding'dir.
Kullanım: `KULLANIM.md`. Entegrasyon: `ENTEGRASYON.md`.
