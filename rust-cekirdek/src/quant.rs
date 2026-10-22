//! Vektör nicemleme (int8) + gömülü "dondurulmuş" paket — MCU/WASM/4GB için.
//!
//! GERÇEK GÖMÜLÜ HİKAYESİ: masaüstünde beyni yaz/eğit (.mimar, f64), sonra DONDUR:
//! vektörleri int8'e nicemle (8× küçük RAM+disk), cevaplarla birlikte tek kompakt
//! bloğa yaz (`QuantPack`). MCU/tarayıcı bu bloğu yükler ve KÜÇÜK no_std-hazır
//! çekirdekle (yalnız `core`+`alloc`; std/dosya/HashMap YOK) en-yakını bulur.
//!
//! Bu modül bilerek SADECE core+alloc kullanır → aynı kaynak no_std bir crate'e
//! kopyalanabilir. (std crate'inde core bir alt-küme olduğu için burada da derlenir.)

extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;

/// Birim vektörü int8'e nicemle: q[i]=round(v[i]/ölçek*127), ölçek=max|v|.
pub fn quantize(v: &[f64]) -> (Vec<i8>, f32) {
    let mut scale = 0.0f64;
    for &x in v { let a = if x < 0.0 { -x } else { x }; if a > scale { scale = a; } }
    if scale == 0.0 { return (alloc::vec![0i8; v.len()], 1.0); }
    let q = v.iter().map(|&x| {
        let s = (x / scale * 127.0).round();
        if s > 127.0 { 127 } else if s < -128.0 { -128 } else { s as i8 }
    }).collect();
    (q, scale as f32)
}

/// İki nicemli vektörün kosinüsü (birim vektörler → nokta çarpımı). Tamsayı iç çarpım
/// + tek ölçek çarpımı (MCU'da hızlı; FPU şart değil, i32 toplam).
pub fn cosine_q(a: &[i8], sa: f32, b: &[i8], sb: f32) -> f64 {
    let n = a.len().min(b.len());
    let mut acc: i64 = 0;
    for i in 0..n { acc += a[i] as i64 * b[i] as i64; }
    acc as f64 * (sa as f64 * sb as f64) / (127.0 * 127.0)
}

/// Dondurulmuş gömülü paket: nicemli vektörler + cevaplar. Tek blok, self-contained.
/// Yalnız core+alloc → no_std çekirdekte çalışır.
pub struct QuantPack {
    pub dim: usize,
    pub vecs: Vec<i8>,        // düz: n*dim
    pub scales: Vec<f32>,     // n
    pub answers: Vec<String>, // n
}

impl QuantPack {
    pub fn new(dim: usize) -> Self {
        QuantPack { dim, vecs: Vec::new(), scales: Vec::new(), answers: Vec::new() }
    }
    pub fn len(&self) -> usize { self.scales.len() }

    pub fn add(&mut self, v: &[f64], answer: &str) {
        let (q, s) = quantize(v);
        // dim'e pad/truncate (homojenlik)
        let mut q = q;
        q.resize(self.dim, 0);
        self.vecs.extend_from_slice(&q);
        self.scales.push(s);
        self.answers.push(String::from(answer));
    }

    /// En-yakın (no_std çekirdek): nicemli sorgu vektörüyle en yüksek kosinüs.
    pub fn nearest(&self, qv: &[f64]) -> Option<(usize, f64)> {
        if self.len() == 0 { return None; }
        let (qq, qs) = quantize(qv);
        let mut qq = qq; qq.resize(self.dim, 0);
        let mut best = (0usize, f64::MIN);
        for i in 0..self.len() {
            let off = i * self.dim;
            let sim = cosine_q(&qq, qs, &self.vecs[off..off + self.dim], self.scales[i]);
            if sim > best.1 { best = (i, sim); }
        }
        Some(best)
    }

    /// Kompakt ikili blok: MAGIC "MQZ1" + dim + n + scales + vecs + cevaplar.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"MQZ1");
        b.extend_from_slice(&(self.dim as u32).to_le_bytes());
        b.extend_from_slice(&(self.len() as u32).to_le_bytes());
        for &s in &self.scales { b.extend_from_slice(&s.to_le_bytes()); }
        for &q in &self.vecs { b.push(q as u8); }
        for a in &self.answers {
            let bytes = a.as_bytes();
            b.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            b.extend_from_slice(bytes);
        }
        b
    }

    pub fn from_bytes(data: &[u8]) -> Option<QuantPack> {
        if data.len() < 12 || &data[0..4] != b"MQZ1" { return None; }
        let dim = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        let n = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let mut off = 12;
        let mut scales = Vec::with_capacity(n);
        for _ in 0..n {
            if off + 4 > data.len() { return None; }
            scales.push(f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]));
            off += 4;
        }
        let vlen = n * dim;
        if off + vlen > data.len() { return None; }
        let vecs: Vec<i8> = data[off..off + vlen].iter().map(|&b| b as i8).collect();
        off += vlen;
        let mut answers = Vec::with_capacity(n);
        for _ in 0..n {
            if off + 4 > data.len() { return None; }
            let l = u32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]) as usize;
            off += 4;
            if off + l > data.len() { return None; }
            answers.push(String::from_utf8_lossy(&data[off..off + l]).into_owned());
            off += l;
        }
        Some(QuantPack { dim, vecs, scales, answers })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed;

    fn unit(seed: &mut u64, dim: usize) -> Vec<f64> {
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim { let mut x=*seed; x^=x>>12; x^=x<<25; x^=x>>27; *seed=x;
            v.push(((x>>11) as f64/(1u64<<53) as f64)*2.0-1.0); }
        let n: f64 = v.iter().map(|x| x*x).sum::<f64>().sqrt();
        for x in v.iter_mut() { *x /= n; }
        v
    }

    #[test]
    fn quant_preserves_cosine() {
        let dim = 256; let mut seed = 7u64;
        let mut maxerr = 0.0f64;
        for _ in 0..200 {
            let a = unit(&mut seed, dim); let b = unit(&mut seed, dim);
            let (qa, sa) = quantize(&a); let (qb, sb) = quantize(&b);
            let exact = embed::cosine(&a, &b);
            let q = cosine_q(&qa, sa, &qb, sb);
            maxerr = maxerr.max((exact - q).abs());
        }
        println!("int8 nicemleme maks kosinüs hatası: {:.4}", maxerr);
        assert!(maxerr < 0.02, "nicemleme hatası çok yüksek: {}", maxerr);
    }

    #[test]
    fn quant_8x_smaller_and_ranking_preserved() {
        // f64 = 8 bayt, i8 = 1 bayt → 8× küçük (kayıt başı dim×7 bayt tasarruf).
        assert_eq!(core::mem::size_of::<f64>() / core::mem::size_of::<i8>(), 8);
        let dim = 128; let mut seed = 99u64;
        let mut pack = QuantPack::new(dim);
        let data: Vec<Vec<f64>> = (0..300).map(|_| unit(&mut seed, dim)).collect();
        for (i, v) in data.iter().enumerate() { pack.add(v, &alloc::format!("cevap{}", i)); }
        // sorgu = 42'nin gürültülü hali → en yakın 42 olmalı
        let mut q = data[42].clone();
        for k in 0..dim { q[k] += 0.02 * ((k % 5) as f64 - 2.0); }
        let nn: f64 = q.iter().map(|z| z*z).sum::<f64>().sqrt();
        for z in q.iter_mut() { *z /= nn; }
        let (idx, sim) = pack.nearest(&q).unwrap();
        assert_eq!(idx, 42, "nicemli en-yakın doğru olmalı (sim={})", sim);
    }

    #[test]
    fn quantpack_roundtrip() {
        let dim = 64; let mut seed = 3u64;
        let mut p = QuantPack::new(dim);
        p.add(&unit(&mut seed, dim), "bir");
        p.add(&unit(&mut seed, dim), "iki");
        let bytes = p.to_bytes();
        let p2 = QuantPack::from_bytes(&bytes).expect("geri yükleme");
        assert_eq!(p2.len(), 2);
        assert_eq!(p2.answers[1], "iki");
        assert_eq!(p2.dim, dim);
    }
}
