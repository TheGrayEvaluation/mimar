//! Hashing embedder + anahtar — Python `selector.py`/`brain._key` ile BİREBİR
//! parite (md5 özellik-hashleme, sha1 anahtar). Böylece Rust ve Python aynı
//! .mimar dosyasını üretir/okur.

use crate::hash::{hex, md5, sha1};

pub const DIM: usize = 256;

pub fn words(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            for l in ch.to_lowercase() { cur.push(l); }
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() { out.push(cur); }
    out
}

fn char_ngrams(ws: &[String], n: usize) -> Vec<String> {
    let t: String = format!("·{}·", ws.join(" "));
    let chars: Vec<char> = t.chars().collect();
    if chars.len() < n { return Vec::new(); }
    (0..=chars.len() - n).map(|i| chars[i..i + n].iter().collect()).collect()
}

fn features(text: &str) -> Vec<String> {
    let ws = words(text);
    let mut f: Vec<String> = ws.iter().map(|w| format!("w:{}", w)).collect();
    for n in [3usize, 4] {
        for g in char_ngrams(&ws, n) {
            f.push(format!("c{}:{}", n, g));
        }
    }
    f
}

fn h(feature: &str) -> (usize, f64) {
    let d = md5(feature.as_bytes());
    let idx = u32::from_le_bytes([d[0], d[1], d[2], d[3]]) as usize % DIM;
    let sign = if d[4] & 1 == 1 { 1.0 } else { -1.0 };
    (idx, sign)
}

pub fn embed(text: &str) -> Vec<f64> {
    let mut v = vec![0.0f64; DIM];
    for f in features(text) {
        let (idx, sign) = h(&f);
        v[idx] += sign;
    }
    normalize(&mut v);
    v
}

pub fn normalize(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() { *x /= norm; }
    }
}

pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Niyet anahtarı: Python _key = sha1(" ".join(tok)).hexdigest()[:16]
pub fn key(text: &str) -> String {
    let norm = words(text).join(" ");
    hex(&sha1(norm.as_bytes()))[..16].to_string()
}
