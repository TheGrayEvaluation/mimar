//! HNSW — Hierarchical Navigable Small World (grafik-tabanlı ANN).
//! LSH'in düşük recall'ını çözer: yüksek recall + log ölçekli arama. Saf std,
//! sıfır bağımlılık (FAISS/Qdrant'ın gömülebilir karşılığı). Kosinüs benzerliği
//! (vektörler L2-normalize → nokta çarpımı); "yakın" = yüksek benzerlik.
//!
//! Arayüz store ile uyumlu: `add(key, vec)` artımlı, `candidates(q) -> HashSet<key>`
//! (en yakın ef aday). Nihai karar store'un güvenlik katmanında (kalibrasyon+marj).

use std::collections::{BinaryHeap, HashSet};

#[derive(Clone, Copy)]
struct Cand { sim: f64, id: u32 }
impl PartialEq for Cand { fn eq(&self, o: &Self) -> bool { self.sim == o.sim } }
impl Eq for Cand {}
impl PartialOrd for Cand { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering { self.sim.total_cmp(&o.sim) }
}

struct Node {
    vec: Vec<f64>,
    key: String,
    neigh: Vec<Vec<u32>>,   // [layer] -> komşu id'leri
}

pub struct Hnsw {
    m: usize,         // katman başı hedef komşu
    m0: usize,        // 0. katmanda (2*m)
    ef_c: usize,      // inşa genişliği
    ef_s: usize,      // arama genişliği
    ml: f64,          // katman normalizasyonu = 1/ln(m)
    nodes: Vec<Node>,
    entry: Option<u32>,
    max_layer: usize,
    rng: u64,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut s = 0.0;
    for i in 0..n { s += a[i] * b[i]; }
    s
}

impl Hnsw {
    pub fn new() -> Self {
        let m = 16usize;
        Hnsw { m, m0: m * 2, ef_c: 200, ef_s: 64,
               ml: 1.0 / (m as f64).ln(),
               nodes: Vec::new(), entry: None, max_layer: 0, rng: 0x9E37_79B9_7F4A_7C15 }
    }

    pub fn len(&self) -> usize { self.nodes.len() }

    fn next_f64(&mut self) -> f64 {
        let mut x = self.rng; x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.rng = x;
        ((x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn rand_level(&mut self) -> usize {
        let r = self.next_f64().max(1e-12);
        (-r.ln() * self.ml).floor() as usize
    }

    fn sim(&self, q: &[f64], id: u32) -> f64 { dot(q, &self.nodes[id as usize].vec) }

    /// Tek katmanda ep'ten en yakına açgözlü in (ef=1).
    fn greedy(&self, q: &[f64], mut ep: u32, layer: usize) -> u32 {
        let mut best = self.sim(q, ep);
        loop {
            let mut improved = false;
            for &nb in &self.nodes[ep as usize].neigh[layer] {
                let s = self.sim(q, nb);
                if s > best { best = s; ep = nb; improved = true; }
            }
            if !improved { break; }
        }
        ep
    }

    /// Bir katmanda ef en yakını bul (standart HNSW arama).
    fn search_layer(&self, q: &[f64], eps: &[u32], ef: usize, layer: usize) -> Vec<Cand> {
        let mut visited: HashSet<u32> = HashSet::new();
        let mut cand: BinaryHeap<Cand> = BinaryHeap::new();            // max-sim (en yakın pop)
        let mut w: BinaryHeap<std::cmp::Reverse<Cand>> = BinaryHeap::new(); // min-sim (en kötü pop)
        for &e in eps {
            let s = self.sim(q, e);
            visited.insert(e);
            cand.push(Cand { sim: s, id: e });
            w.push(std::cmp::Reverse(Cand { sim: s, id: e }));
        }
        while let Some(c) = cand.pop() {
            let worst = w.peek().map(|r| r.0.sim).unwrap_or(f64::MIN);
            if c.sim < worst && w.len() >= ef { break; }
            for &nb in &self.nodes[c.id as usize].neigh[layer] {
                if visited.insert(nb) {
                    let s = self.sim(q, nb);
                    let worst = w.peek().map(|r| r.0.sim).unwrap_or(f64::MIN);
                    if s > worst || w.len() < ef {
                        cand.push(Cand { sim: s, id: nb });
                        w.push(std::cmp::Reverse(Cand { sim: s, id: nb }));
                        if w.len() > ef { w.pop(); }
                    }
                }
            }
        }
        let mut out: Vec<Cand> = w.into_iter().map(|r| r.0).collect();
        out.sort_by(|a, b| b.sim.total_cmp(&a.sim));   // yüksek benzerlik önce
        out
    }

    pub fn add(&mut self, key: &str, vec: &[f64]) {
        let id = self.nodes.len() as u32;
        let l = self.rand_level();
        self.nodes.push(Node { vec: vec.to_vec(), key: key.to_string(),
                               neigh: vec![Vec::new(); l + 1] });
        let entry = match self.entry {
            None => { self.entry = Some(id); self.max_layer = l; return; }
            Some(e) => e,
        };
        let q = vec;
        let mut ep = entry;
        // Üst katmanlardan hedef katmana açgözlü in
        let mut lc = self.max_layer;
        while lc > l {
            ep = self.greedy(q, ep, lc);
            lc -= 1;
        }
        // l'den 0'a kadar bağla
        let mut lc = l.min(self.max_layer) as i64;
        while lc >= 0 {
            let layer = lc as usize;
            let w = self.search_layer(q, &[ep], self.ef_c, layer);
            let m = if layer == 0 { self.m0 } else { self.m };
            let selected: Vec<u32> = w.iter().take(m).map(|c| c.id).collect();
            for &nb in &selected {
                self.nodes[id as usize].neigh[layer].push(nb);
                self.nodes[nb as usize].neigh[layer].push(id);
                // komşunun listesi taşarsa en yakınlara buda
                if self.nodes[nb as usize].neigh[layer].len() > m {
                    self.prune(nb, layer, m);
                }
            }
            if let Some(first) = w.first() { ep = first.id; }
            lc -= 1;
        }
        if l > self.max_layer { self.max_layer = l; self.entry = Some(id); }
    }

    fn prune(&mut self, id: u32, layer: usize, m: usize) {
        let base = self.nodes[id as usize].vec.clone();
        let mut scored: Vec<Cand> = self.nodes[id as usize].neigh[layer].iter()
            .map(|&nb| Cand { sim: dot(&base, &self.nodes[nb as usize].vec), id: nb })
            .collect();
        scored.sort_by(|a, b| b.sim.total_cmp(&a.sim));
        scored.truncate(m);
        self.nodes[id as usize].neigh[layer] = scored.into_iter().map(|c| c.id).collect();
    }

    /// En yakın adayların anahtar kümesi (store rerank + güvenlik katmanı için).
    pub fn candidates(&self, q: &[f64]) -> HashSet<String> {
        let mut out = HashSet::new();
        let ep = match self.entry { Some(e) => e, None => return out };
        let mut ep = ep;
        let mut lc = self.max_layer;
        while lc > 0 { ep = self.greedy(q, ep, lc); lc -= 1; }
        for c in self.search_layer(q, &[ep], self.ef_s, 0) {
            out.insert(self.nodes[c.id as usize].key.clone());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd_vec(seed: &mut u64, dim: usize) -> Vec<f64> {
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            let mut x = *seed; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; *seed = x;
            v.push(((x >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0);
        }
        let n: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in v.iter_mut() { *x /= n; }
        v
    }

    #[test]
    fn recall_vs_brute() {
        let dim = 64; let n = 2000;
        let mut seed = 12345u64;
        let data: Vec<Vec<f64>> = (0..n).map(|_| rnd_vec(&mut seed, dim)).collect();
        let mut h = Hnsw::new();
        for (i, v) in data.iter().enumerate() { h.add(&format!("k{}", i), v); }
        // Sorgu: mevcut vektörün hafif perturbe hali → gerçek en-yakın = kaynak
        let mut hit = 0; let q = 100;
        for _ in 0..q {
            let mut x = seed; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; seed = x;
            let idx = (x % n as u64) as usize;
            let mut query = data[idx].clone();
            // küçük gürültü
            for k in 0..dim { let mut y=seed; y^=y>>12; y^=y<<25; y^=y>>27; seed=y;
                query[k] += ((y>>11) as f64/(1u64<<53) as f64 - 0.5)*0.05; }
            let nn: f64 = query.iter().map(|z| z*z).sum::<f64>().sqrt();
            for z in query.iter_mut() { *z /= nn; }
            // brute en yakın
            let mut bi = 0; let mut bs = -2.0;
            for (j, v) in data.iter().enumerate() { let s = dot(&query, v); if s>bs {bs=s; bi=j;} }
            if h.candidates(&query).contains(&format!("k{}", bi)) { hit += 1; }
        }
        let recall = hit as f64 / q as f64;
        println!("HNSW recall@aday: {:.2} ({}/{})", recall, hit, q);
        assert!(recall >= 0.90, "HNSW recall düşük: {}", recall);
    }

    #[test]
    fn hnsw_beats_lsh_recall() {
        use crate::index::Lsh;
        use std::time::Instant;
        let dim = 256; let n = 3000;
        let mut seed = 999u64;
        let data: Vec<Vec<f64>> = (0..n).map(|_| rnd_vec(&mut seed, dim)).collect();
        let mut h = Hnsw::new();
        let mut l = Lsh::new(dim);
        let t0 = Instant::now();
        for (i, v) in data.iter().enumerate() { h.add(&format!("k{}", i), v); }
        let t_h = t0.elapsed();
        let t1 = Instant::now();
        for (i, v) in data.iter().enumerate() { l.add(&format!("k{}", i), v); }
        let t_l = t1.elapsed();

        let q = 200;
        let (mut hit_h, mut hit_l, mut sz_h, mut sz_l) = (0, 0, 0usize, 0usize);
        let (mut ms_h, mut ms_l) = (0u128, 0u128);
        for _ in 0..q {
            let mut x = seed; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; seed = x;
            let idx = (x % n as u64) as usize;
            let mut query = data[idx].clone();
            // BÜYÜK gürültü → en-yakın artık birebir kopya değil, gerçekten "zor" komşu
            for k in 0..dim { let mut y=seed; y^=y>>12; y^=y<<25; y^=y>>27; seed=y;
                query[k] += ((y>>11) as f64/(1u64<<53) as f64 - 0.5)*0.9; }
            let nn: f64 = query.iter().map(|z| z*z).sum::<f64>().sqrt();
            for z in query.iter_mut() { *z /= nn; }
            let mut bi = 0; let mut bs = -2.0;
            for (j, v) in data.iter().enumerate() { let s = dot(&query, v); if s>bs {bs=s; bi=j;} }
            let want = format!("k{}", bi);
            let a = Instant::now(); let ch = h.candidates(&query); ms_h += a.elapsed().as_micros();
            let b = Instant::now(); let cl = l.candidates(&query); ms_l += b.elapsed().as_micros();
            sz_h += ch.len(); sz_l += cl.len();
            if ch.contains(&want) { hit_h += 1; }
            if cl.contains(&want) { hit_l += 1; }
        }
        let (rh, rl) = (hit_h as f64/q as f64, hit_l as f64/q as f64);
        println!("--- N={} dim={} ---", n, dim);
        println!("HNSW: recall={:.2} aday~{} inşa={:?} sorgu~{}µs",
                 rh, sz_h/q, t_h, ms_h/q as u128);
        println!("LSH : recall={:.2} aday~{} inşa={:?} sorgu~{}µs",
                 rl, sz_l/q, t_l, ms_l/q as u128);
        assert!(rh >= rl, "HNSW recall LSH'ten düşük olmamalı: {} < {}", rh, rl);
        assert!(rh >= 0.90, "HNSW recall zor komşuda bile yüksek olmalı: {}", rh);
    }
}
