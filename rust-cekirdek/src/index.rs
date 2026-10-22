//! LSH anlamsal indeks (rastgele hiper-düzlem) — saf std, zero-dep PRNG.
//! Lineer taramayı alt-lineere indirir. Yaklaşıktır; birebir/öğretilmiş eşleşme
//! store'da ayrıca anahtarla kesin bulunur (güvenlik).

use std::collections::{HashMap, HashSet};

const TAU: f64 = std::f64::consts::PI * 2.0;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }
}

pub struct Lsh {
    dim: usize,
    l: usize,
    k: usize,
    planes: Vec<Vec<Vec<f64>>>,          // [tablo][bit][dim]
    tables: Vec<HashMap<u64, Vec<String>>>,
    pub indexed: usize,
}

impl Lsh {
    pub fn new(dim: usize) -> Self {
        let (l, k) = (16usize, 10usize);
        let mut rng = Rng(0x1234_5678_9abc_def1);
        let planes = (0..l).map(|_| {
            (0..k).map(|_| (0..dim).map(|_| rng.gauss()).collect()).collect()
        }).collect();
        Lsh { dim, l, k, planes, tables: vec![HashMap::new(); l], indexed: 0 }
    }

    fn sig(&self, vec: &[f64], t: usize) -> u64 {
        let mut s = 0u64;
        for i in 0..self.k {
            let dot: f64 = self.planes[t][i].iter().zip(vec).map(|(a, b)| a * b).sum();
            if dot >= 0.0 { s |= 1 << i; }
        }
        s
    }

    pub fn add(&mut self, key: &str, vec: &[f64]) {
        if vec.len() != self.dim { return; }
        for t in 0..self.l {
            let sig = self.sig(vec, t);
            self.tables[t].entry(sig).or_default().push(key.to_string());
        }
        self.indexed += 1;
    }

    pub fn candidates(&self, vec: &[f64]) -> HashSet<String> {
        let mut c = HashSet::new();
        for t in 0..self.l {
            if let Some(keys) = self.tables[t].get(&self.sig(vec, t)) {
                for k in keys { c.insert(k.clone()); }
            }
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed;

    #[test]
    fn finds_near_duplicate() {
        let mut lsh = Lsh::new(embed::DIM);
        let texts: Vec<String> = (0..300).map(|i| format!("benzersiz konu sayi {}", i)).collect();
        for (i, t) in texts.iter().enumerate() {
            lsh.add(&format!("k{}", i), &embed::embed(t));
        }
        // Birebir vektör kendi kovasında bulunmalı
        let q = embed::embed(&texts[42]);
        assert!(lsh.candidates(&q).contains("k42"), "birebir vektör adaylarda olmalı");
        // Aday sayısı toplamın küçük bir kısmı (alt-lineer)
        assert!(lsh.candidates(&q).len() < 300, "adaylar tümünden az olmalı");
    }
}
