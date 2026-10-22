//! At-rest şifreleme — .mimar beynini parolayla şifrele (gizlilik + bütünlük).
//! encrypt-then-MAC: SHA-256 CTR anahtar akışı + HMAC-SHA256. SIFIR bağımlılık
//! (mevcut hash modülü). Anahtar türetme: tuz + 4096 tur SHA-256 (brute yavaşlatma).
//!
//! DÜRÜST SINIR: bu AES-GCM/libsodium DEĞİL; SHA-256 tabanlı sağlam bir yapı
//! (CTR+HMAC). Standart/mevzuat gereken yerde AES-GCM önerilir. Ama gerçek
//! gizlilik+tamper-tespiti sağlar ve hiçbir dış crate gerektirmez.

use crate::hash::{sha256, hmac_sha256};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const MENC: &[u8; 4] = b"MENC";
static CTR: AtomicU64 = AtomicU64::new(0);

fn gen_salt() -> [u8; 16] {
    let t = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let c = CTR.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id() as u64;
    let mut b = Vec::new();
    b.extend_from_slice(&t.to_le_bytes());
    b.extend_from_slice(&c.to_le_bytes());
    b.extend_from_slice(&pid.to_le_bytes());
    let h = sha256(&b);
    let mut s = [0u8; 16];
    s.copy_from_slice(&h[..16]);
    s
}

/// Anahtar türetme: K0=SHA256(pass||salt), sonra 4096 tur K=SHA256(K||pass||salt).
fn derive_key(pass: &[u8], salt: &[u8]) -> [u8; 32] {
    let mut buf = Vec::with_capacity(pass.len() + salt.len());
    buf.extend_from_slice(pass);
    buf.extend_from_slice(salt);
    let mut k = sha256(&buf);
    for _ in 0..4096 {
        let mut b2 = Vec::with_capacity(32 + pass.len() + salt.len());
        b2.extend_from_slice(&k);
        b2.extend_from_slice(pass);
        b2.extend_from_slice(salt);
        k = sha256(&b2);
    }
    k
}

/// CTR anahtar akışı: blok_i = SHA256(key || i_be); XOR.
fn apply_keystream(key: &[u8; 32], data: &mut [u8]) {
    let mut counter: u64 = 0;
    let mut off = 0;
    while off < data.len() {
        let mut b = Vec::with_capacity(40);
        b.extend_from_slice(key);
        b.extend_from_slice(&counter.to_be_bytes());
        let ks = sha256(&b);
        let n = (data.len() - off).min(32);
        for j in 0..n { data[off + j] ^= ks[j]; }
        off += 32;
        counter += 1;
    }
}

/// Sabit-zamanlı bayt karşılaştırma (MAC doğrulama — zamanlama sızıntısı yok).
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() { return false; }
    let mut diff = 0u8;
    for i in 0..a.len() { diff |= a[i] ^ b[i]; }
    diff == 0
}

/// Şifrele: MENC(4) + tuz(16) + mac_hex(64) + şifreli_metin.
pub fn encrypt(plain: &[u8], pass: &str) -> Vec<u8> {
    let salt = gen_salt();
    let key = derive_key(pass.as_bytes(), &salt);
    let mut ct = plain.to_vec();
    apply_keystream(&key, &mut ct);
    let mut maced = Vec::with_capacity(16 + ct.len());
    maced.extend_from_slice(&salt);
    maced.extend_from_slice(&ct);
    let mac = hmac_sha256(&key, &maced);   // 64 hex char
    let mut out = Vec::with_capacity(4 + 16 + 64 + ct.len());
    out.extend_from_slice(MENC);
    out.extend_from_slice(&salt);
    out.extend_from_slice(mac.as_bytes());
    out.extend_from_slice(&ct);
    out
}

/// Çöz: MAC doğrula (yanlış parola/bozulma → Err), sonra deşifre.
pub fn decrypt(data: &[u8], pass: &str) -> Result<Vec<u8>, String> {
    if data.len() < 4 + 16 + 64 || &data[0..4] != MENC {
        return Err("şifreli Mimar bloğu değil (MENC yok / çok kısa)".into());
    }
    let salt = &data[4..20];
    let mac_hex = &data[20..84];
    let ct = &data[84..];
    let key = derive_key(pass.as_bytes(), salt);
    let mut maced = Vec::with_capacity(16 + ct.len());
    maced.extend_from_slice(salt);
    maced.extend_from_slice(ct);
    let expect = hmac_sha256(&key, &maced);
    if !ct_eq(expect.as_bytes(), mac_hex) {
        return Err("kimlik doğrulama başarısız (yanlış parola veya dosya bozulmuş/kurcalanmış)".into());
    }
    let mut pt = ct.to_vec();
    apply_keystream(&key, &mut pt);
    Ok(pt)
}

/// Blok şifreli mi (MENC başlığı)?
pub fn is_encrypted(data: &[u8]) -> bool {
    data.len() >= 4 && &data[0..4] == MENC
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let plain = b"MIMR gizli beyin icerigi: iade=14 gun, sifre=1234";
        let enc = encrypt(plain, "parola-guclu");
        assert!(is_encrypted(&enc));
        assert_ne!(&enc[84..], &plain[..], "şifreli metin düzden farklı olmalı");
        let dec = decrypt(&enc, "parola-guclu").unwrap();
        assert_eq!(dec, plain);
    }

    #[test]
    fn wrong_password_fails() {
        let enc = encrypt(b"sir", "dogru");
        assert!(decrypt(&enc, "yanlis").is_err(), "yanlış parola reddedilmeli");
    }

    #[test]
    fn tamper_detected() {
        let mut enc = encrypt(b"onemli veri blogu", "p");
        let n = enc.len();
        enc[n - 1] ^= 0xFF;   // şifreli metni kurcala
        assert!(decrypt(&enc, "p").is_err(), "kurcalama MAC ile yakalanmalı");
    }

    #[test]
    fn salt_makes_outputs_differ() {
        let a = encrypt(b"ayni girdi", "p");
        let b = encrypt(b"ayni girdi", "p");
        assert_ne!(a, b, "tuz → aynı düz metin farklı şifreli çıktı (determinizm yok)");
        // ama ikisi de doğru çözülür
        assert_eq!(decrypt(&a, "p").unwrap(), decrypt(&b, "p").unwrap());
    }
}
