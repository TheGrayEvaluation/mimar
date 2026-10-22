//! Gerçek makine öğrenmesi — motorun içinde.
//!
//! Nearest-centroid SIĞdı (kullanıcı haklı). Burada GRADYANLA ÖĞRENEN bir model
//! var: tek-vs-hepsi (one-vs-rest) ONLINE LOJİSTİK REGRESYON (SGD). Her niyet
//! bir sınıftır; her sınıfın bir ağırlık vektörü w + bias b vardır. Örnek
//! geldikçe (teach/öğren) ağırlıklar gradyan adımıyla güncellenir — yani model
//! VERİDEN ÖĞRENİR, sadece hatırlamaz. Held-out doğruluğu eğitimle artar (test).
//!
//! Saf std, sıfır bağımlılık → gömülüye taşınabilir. Ağırlıklar beyin dosyasında
//! yaşar (öğrenilen model .mimar içinde, SQLite'ın veriyi tutması gibi).

use crate::json::J;

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub struct Cls {
    pub label: String,
    pub w: Vec<f64>,
    pub b: f64,
}

pub struct LogReg {
    pub dim: usize,
    pub lr: f64,
    pub classes: Vec<Cls>,
}

impl LogReg {
    pub fn new(dim: usize, lr: f64) -> Self {
        LogReg { dim, lr, classes: Vec::new() }
    }

    fn idx(&mut self, label: &str) -> usize {
        if let Some(i) = self.classes.iter().position(|c| c.label == label) {
            return i;
        }
        self.classes.push(Cls { label: label.to_string(), w: vec![0.0; self.dim], b: 0.0 });
        self.classes.len() - 1
    }

    /// Bir (örnek, etiket) çiftiyle ONLINE eğit (OvR SGD). Tüm sınıflara negatif,
    /// hedef sınıfa pozitif gradyan uygular → gerçek öğrenme.
    pub fn train(&mut self, x: &[f64], label: &str) {
        self.idx(label);
        let lr = self.lr;
        for c in self.classes.iter_mut() {
            let t = if c.label == label { 1.0 } else { 0.0 };
            let p = sigmoid(dot(&c.w, x) + c.b);
            let e = t - p;                 // türev: (hedef - tahmin)
            for k in 0..c.w.len() {
                c.w[k] += lr * e * x[k];
            }
            c.b += lr * e;
        }
    }

    /// En yüksek olasılıklı sınıfı (etiket, olasılık) döndür.
    pub fn predict(&self, x: &[f64]) -> Option<(String, f64)> {
        let mut best: Option<(String, f64)> = None;
        for c in &self.classes {
            let p = sigmoid(dot(&c.w, x) + c.b);
            if best.as_ref().map_or(true, |b| p > b.1) {
                best = Some((c.label.clone(), p));
            }
        }
        best
    }

    // ---- beyin dosyasında saklama ----
    pub fn to_json(&self) -> J {
        let classes = self.classes.iter().map(|c| {
            J::Obj(vec![
                ("label".into(), J::Str(c.label.clone())),
                ("b".into(), J::Num(c.b)),
                ("w".into(), J::Arr(c.w.iter().map(|x| J::Num(*x)).collect())),
            ])
        }).collect();
        J::Obj(vec![
            ("dim".into(), J::Num(self.dim as f64)),
            ("lr".into(), J::Num(self.lr)),
            ("classes".into(), J::Arr(classes)),
        ])
    }

    pub fn from_json(j: &J) -> LogReg {
        let dim = j.get("dim").and_then(|v| v.as_f64()).unwrap_or(256.0) as usize;
        let lr = j.get("lr").and_then(|v| v.as_f64()).unwrap_or(0.3);
        let mut m = LogReg::new(dim, lr);
        if let Some(J::Arr(cs)) = j.get("classes") {
            for c in cs {
                let label = c.get("label").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let b = c.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let w = match c.get("w") {
                    Some(J::Arr(a)) => a.iter().filter_map(|x| x.as_f64()).collect(),
                    _ => vec![0.0; dim],
                };
                m.classes.push(Cls { label, w, b });
            }
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed;

    // Üç sınıf, ayrık kelime dağarcıkları. Eğitim ve HELD-OUT test metinleri
    // farklı; model ezberleyemez, GENELLEMELİ.
    fn data() -> (Vec<(&'static str, &'static str)>, Vec<(&'static str, &'static str)>) {
        let train = vec![
            ("futbol maçı gol", "spor"), ("basketbol smaç pota", "spor"),
            ("tenis raket servis", "spor"), ("koşu maraton atlet", "spor"),
            ("çorba kebap tatlı", "yemek"), ("makarna sos peynir", "yemek"),
            ("pilav fasulye turşu", "yemek"), ("salata zeytinyağı limon", "yemek"),
            ("atom elektron proton", "bilim"), ("hücre dna genetik", "bilim"),
            ("gezegen yıldız galaksi", "bilim"), ("enerji kütle hız", "bilim"),
        ];
        let test = vec![
            ("gol pota raket", "spor"), ("maraton smaç servis", "spor"),
            ("kebap makarna salata", "yemek"), ("peynir turşu tatlı", "yemek"),
            ("elektron dna gezegen", "bilim"), ("genetik kütle yıldız", "bilim"),
        ];
        (train, test)
    }

    fn accuracy(m: &LogReg, test: &[(&str, &str)]) -> f64 {
        let mut ok = 0;
        for (txt, lbl) in test {
            let x = embed::embed(txt);
            if m.predict(&x).map(|(l, _)| l) == Some(lbl.to_string()) {
                ok += 1;
            }
        }
        ok as f64 / test.len() as f64
    }

    #[test]
    fn genuinely_learns() {
        let (train, test) = data();
        let mut m = LogReg::new(embed::DIM, 0.5);
        // 1 epoch sonrası
        for (txt, lbl) in &train {
            m.train(&embed::embed(txt), lbl);
        }
        let acc1 = accuracy(&m, &test);
        // 25 epoch daha
        for _ in 0..25 {
            for (txt, lbl) in &train {
                m.train(&embed::embed(txt), lbl);
            }
        }
        let acc25 = accuracy(&m, &test);
        println!("held-out doğruluk: 1 epoch={:.2}  26 epoch={:.2}", acc1, acc25);
        assert!(acc25 >= 0.8, "eğitimle doğruluk yükselmeli (acc25={})", acc25);
        assert!(acc25 >= acc1, "öğrenme doğruluğu düşürmemeli");
    }

    #[test]
    fn persist_roundtrip() {
        let mut m = LogReg::new(embed::DIM, 0.4);
        m.train(&embed::embed("futbol gol"), "spor");
        let j = m.to_json();
        let m2 = LogReg::from_json(&j);
        let x = embed::embed("futbol gol");
        assert_eq!(m.predict(&x).unwrap().0, m2.predict(&x).unwrap().0);
    }
}
