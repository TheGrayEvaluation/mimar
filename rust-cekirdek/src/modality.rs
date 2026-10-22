//! Çok-kiplilik — Mimar vektör-AGNOSTİKTİR: bir kip bir vektöre parmaklanabiliyorsa
//! AYNI store/indeks/güvenlik/damıtma makinesi değişmeden çalışır. Burada üç NATIVE,
//! sıfır-dep, DETERMİNİSTİK parmak izi embedder'ı var (hepsi 256 boyut = metin uzayıyla
//! aynı şekil, böylece Brain/LSH→HNSW/ML dokunulmadan kullanır):
//!   • DataEmbedder  — yapısal kayıt (key=value) özellik-hash'i (Embedder; uçtan uca).
//!   • embed_image   — algısal hash (16×16 ortalama-havuz gri piksel) → yakın-kopya görsel.
//!   • embed_audio   — kaba güç spektrumu (DFT binleme) → yakın-kopya ses.
//!
//! İLKE "Mimar yönlendirir, host çalıştırır": host görseli/sesi ÇÖZER (her dilde
//! decoder var), Mimar ham piksel/örneği parmaklar. GERÇEK ANLAMSAL görü/işitme
//! (CLIP/wav2vec) bge-m3 gibi OPT-IN dış embedder'dır — mimarî onu da kaldırır.

use crate::embed::{self, DIM};
use crate::embedder::Embedder;

// ---- DATA (yapısal kayıt) — tam Embedder, uçtan uca ----

/// Kaydı kanonikleştir: alanları sırala, "key=value" birleştir (deterministik).
pub fn canon_record(fields: &[(String, String)]) -> String {
    let mut f = fields.to_vec();
    f.sort_by(|a, b| a.0.cmp(&b.0));
    f.iter().map(|(k, v)| format!("{}={}", k.trim(), v.trim()))
        .collect::<Vec<_>>().join(" ")
}

/// Yapısal veri embedder'ı: her "key=value" alanını ayrı özellik olarak hash'ler
/// (alan düzeyinde ağırlık; metin bag-of-words'ten daha yapısal).
pub struct DataEmbedder;
impl Embedder for DataEmbedder {
    fn embed(&self, text: &str) -> Vec<f64> {
        // text ya kanonik "k=v k=v" ya da serbest; token=alan.
        let mut v = vec![0.0f64; DIM];
        for field in text.split_whitespace() {
            let h = fnv1a(field.as_bytes());
            let idx = (h % DIM as u64) as usize;
            let sign = if (h >> 63) & 1 == 1 { -1.0 } else { 1.0 };
            v[idx] += sign;
            // alan-içi key ayrıca sinyal (değer değişse de key eşleşmesi tutar)
            if let Some((k, _)) = field.split_once('=') {
                let hk = fnv1a(k.as_bytes());
                v[(hk % DIM as u64) as usize] += 0.5;
            }
        }
        embed::normalize(&mut v);
        v
    }
    fn dim(&self) -> usize { DIM }
    fn id(&self) -> String { "data-256".into() }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); }
    h
}

// ---- IMAGE (algısal hash) ----

/// Gri piksel düzlemini (satır-öncelikli, w×h, 0..=255) 16×16 ortalama-havuza indir,
/// düzleştir, normalize et. Yakın-kopya/benzer görseller yüksek kosinüs verir.
pub fn embed_image(gray: &[u8], w: usize, h: usize) -> Vec<f64> {
    let (gw, gh) = (16usize, 16usize);
    let mut v = vec![0.0f64; gw * gh];
    if w == 0 || h == 0 || gray.len() < w * h { return v; }
    for gy in 0..gh {
        for gx in 0..gw {
            // bu ızgara hücresine düşen kaynak blok ortalaması
            let x0 = gx * w / gw; let x1 = ((gx + 1) * w / gw).max(x0 + 1).min(w);
            let y0 = gy * h / gh; let y1 = ((gy + 1) * h / gh).max(y0 + 1).min(h);
            let mut sum = 0u64; let mut cnt = 0u64;
            for y in y0..y1 { for x in x0..x1 {
                sum += gray[y * w + x] as u64; cnt += 1;
            }}
            v[gy * gw + gx] = if cnt > 0 { sum as f64 / cnt as f64 } else { 0.0 };
        }
    }
    // parlaklık-bağımsız: ortalamayı çıkar (kontrast yapısı kalsın), sonra normalize
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    for x in v.iter_mut() { *x -= mean; }
    embed::normalize(&mut v);
    v
}

// ---- AUDIO (kaba güç spektrumu) ----

/// PCM örneklerini (f64, mono, -1..=1) kaba log-güç spektrumuna indir (DFT binleme),
/// normalize et. Yakın-kopya/benzer ses yüksek kosinüs verir.
pub fn embed_audio(samples: &[f64]) -> Vec<f64> {
    let bins = 256usize;
    let mut v = vec![0.0f64; bins];
    if samples.is_empty() { return v; }
    // en fazla 4096 örnek kullan (fingerprint; tam DFT gerekmez)
    let n = samples.len().min(4096);
    let step = samples.len() / n.max(1);
    let s: Vec<f64> = (0..n).map(|i| samples[i * step]).collect();
    let nn = s.len() as f64;
    for (b, slot) in v.iter_mut().enumerate() {
        // bin frekansı: 0..bins → 0..Nyquist
        let freq = b as f64 * 0.5 / bins as f64;   // döngü/örnek
        let w = 2.0 * std::f64::consts::PI * freq;
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (i, &x) in s.iter().enumerate() {
            let ph = w * i as f64;
            re += x * ph.cos();
            im -= x * ph.sin();
        }
        let power = (re * re + im * im).sqrt() / nn;
        *slot = (1.0 + power).ln();   // log-güç (dinamik aralık)
    }
    embed::normalize(&mut v);
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::cosine;

    #[test]
    fn data_matches_similar_records_separates_different() {
        let e = DataEmbedder;
        let a = e.embed(&canon_record(&[("sehir".into(), "Ankara".into()), ("tutar".into(), "500".into())]));
        let b = e.embed(&canon_record(&[("tutar".into(), "500".into()), ("sehir".into(), "Ankara".into())]));
        let c = e.embed(&canon_record(&[("sehir".into(), "Izmir".into()), ("tutar".into(), "9999".into())]));
        assert!(cosine(&a, &b) > 0.999, "alan sırası farkı ETKİLEMEMELİ: {}", cosine(&a, &b));
        assert!(cosine(&a, &c) < cosine(&a, &b), "farklı kayıt daha düşük benzerlik");
    }

    #[test]
    fn image_near_duplicate_high_unrelated_low() {
        // 32×32 gradyan görsel + gürültülü kopyası vs tersine-gradyan
        let (w, h) = (32, 32);
        let grad: Vec<u8> = (0..w*h).map(|i| ((i % w) * 255 / w) as u8).collect();
        let mut noisy = grad.clone();
        for i in (0..noisy.len()).step_by(7) { noisy[i] = noisy[i].saturating_add(10); }
        let inv: Vec<u8> = grad.iter().map(|p| 255 - p).collect();
        let va = embed_image(&grad, w, h);
        let vb = embed_image(&noisy, w, h);
        let vc = embed_image(&inv, w, h);
        assert!(cosine(&va, &vb) > 0.95, "yakın-kopya yüksek: {}", cosine(&va, &vb));
        assert!(cosine(&va, &vc) < cosine(&va, &vb), "ters görsel daha düşük: {} vs {}",
                cosine(&va, &vc), cosine(&va, &vb));
    }

    #[test]
    fn audio_near_duplicate_high_unrelated_low() {
        let sr = 8000.0;
        let tone = |f: f64, n: usize| -> Vec<f64> {
            (0..n).map(|i| (2.0*std::f64::consts::PI*f*i as f64/sr).sin()).collect()
        };
        let a = embed_audio(&tone(440.0, 4000));
        let b = embed_audio(&tone(442.0, 4000));   // neredeyse aynı ton
        let c = embed_audio(&tone(1500.0, 4000));  // çok farklı ton
        assert!(cosine(&a, &b) > 0.9, "yakın ton yüksek: {}", cosine(&a, &b));
        assert!(cosine(&a, &c) < cosine(&a, &b), "farklı ton daha düşük: {} vs {}",
                cosine(&a, &c), cosine(&a, &b));
    }
}
