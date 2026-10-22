# Güvenlik Politikası

## Açık bildirimi
Bir güvenlik açığı bulursan lütfen **herkese açık issue açma**. Bunun yerine
`GUVENLIK_EPOSTASI` (yayın öncesi doldurulacak) adresine özel olarak bildir.
Mümkünse: etkilenen sürüm, yeniden üretim adımları, olası etki.

## Sürüm desteği
Şu an **v0.1 (deneysel/erken)**. Üretim-sınıfı güvence HENÜZ verilmez; API ve
format değişebilir. Kritik işlerde kullanmadan önce kendi değerlendirmeni yap.

## Dürüst güvenlik notları (bilerek şeffaf)
- **At-rest şifreleme** (`encrypt`/`decrypt`): SHA-256 CTR + HMAC (encrypt-then-MAC),
  4096-tur KDF. Gerçek gizlilik+bütünlük sağlar ama **AES-GCM/denetlenmiş bir kütüphane
  değildir.** Mevzuat/yüksek-risk senaryolarında vetted bir AEAD (libsodium, AES-GCM)
  ve bağımsız kripto denetimi önerilir.
- **Prompt-injection taraması** (`scan` / `mimar_scan`): **sezgisel** kalıp tespitidir
  (TR+EN). Yardımcıdır, kesin değildir; atlatılabilir. Tek savunma hattı olarak görme.
- **Sunucu TLS** (`tls` özelliği): rustls tabanlı; kendinden-imzalı sertifika yalnız
  geliştirme içindir. Üretimde gerçek sertifika (`--tls-cert/--tls-key`) ya da ters-proxy.
- **Eşleşme güvenliği**: "benzerlik ≠ aynı cevap" riski birebir-anahtar + kalibrasyon +
  marj-çekimser + geri bildirim ile ele alınır; yine de yeni/benzer verisette eşik ayarı
  (bkz. `calibrate`) gerekebilir.
- **Zehirleme**: `teach` beyne yazar. Güvenilmeyen kaynaktan otomatik öğrenmede
  `policy` (öğrenme kapat) ve `scan` ile ön-filtre kullan.

## Kapsam
Bu bir kütüphanedir; onu gömen uygulamanın genel güvenliği (kimlik doğrulama,
ağ, veri saklama) uygulamanın sorumluluğundadır.
