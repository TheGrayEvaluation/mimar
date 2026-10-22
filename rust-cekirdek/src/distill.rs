//! Davranış damıtma — GERÇEK, ölçülen (mock değil).
//! Fikir: bir modelin (A) davranışını yüzlerce istemle yakala, tek bir `.mimar`'a
//! damıt; o dosya BAŞKA bir modele (B) takıldığında B "aynı şeye dönüşür".
//! Bu "aynılık" EVAL'le ÖLÇÜLÜR, iddia edilmez: B+beyin vs B-yalnız (taban).
//!
//! Neden çalışır: yakalanan istemler beyinden birebir (0ms, modelden BAĞIMSIZ)
//! yeniden üretilir; parafrazlar anlamsal eşleşmeyle. Sadakat bir SPEKTRUMDUR:
//! yakalanan=1.0, parafraz=anlamsal, en yüksek=LoRA (gerçek eğitim ister, dürüst sınır).

use crate::store::Brain;
use crate::engine::Engine;
use crate::embed;

pub struct FidReport {
    pub oran: f64,
    pub uyan: usize,
    pub toplam: usize,
    pub detay: Vec<(String, bool, f64, String)>,  // (girdi, uydu, benzerlik, çıktı)
}

/// İki serbest-metin cevabın "aynılığı": birebir (güçlü sinyal) veya hash-kosinüs eşiği.
fn matches(produced: &str, expected: &str, thresh: f64) -> (bool, f64) {
    if produced.trim() == expected.trim() { return (true, 1.0); }
    let s = embed::cosine(&embed::embed(produced), &embed::embed(expected));
    (s >= thresh, s)
}

/// A modelinin davranışını yakala: her istem → modelin çıktısı → beyne öğret.
/// "Yüzlerce sohbet" bu çiftlerdir: eğitim verisi + regresyon testi aynı anda.
pub fn capture(brain: &mut Brain, engine: &Engine, prompts: &[String], system: &str)
    -> Result<Vec<(String, String)>, String> {
    let mut pairs = Vec::new();
    for p in prompts {
        if p.trim().is_empty() { continue; }
        let out = engine.complete(system, p)?;
        brain.teach(p, &out);
        pairs.push((p.clone(), out));
    }
    brain.save().map_err(|e| e.to_string())?;
    Ok(pairs)
}

/// Beyni motora TAK ve davranış eval'lerini koştur → "aynılık" oranını ÖLÇ.
/// NOT: çağıran brain.save() ÇAĞIRMAZ → ölçüm sırasındaki RAM-içi önbellek atılır
/// (ölçüm beyni kirletmez).
pub fn fidelity(brain: &mut Brain, engine: &Engine, cases: &[(String, String)], thresh: f64)
    -> Result<FidReport, String> {
    let mut hit = 0;
    let mut detay = Vec::new();
    for (inp, expected) in cases {
        let (out, _src, _sim) = brain.ask(inp, engine, false).map_err(|e| e.to_string())?;
        let (ok, s) = matches(&out, expected, thresh);
        if ok { hit += 1; }
        detay.push((inp.clone(), ok, s, out));
    }
    Ok(FidReport {
        oran: if cases.is_empty() { 0.0 } else { hit as f64 / cases.len() as f64 },
        uyan: hit, toplam: cases.len(), detay,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Brain;
    use crate::engine::Engine;

    #[test]
    fn captured_behavior_reproduces_model_independently() {
        // Davranış yakalanmış (teach) niyetler, HANGİ model takılırsa takılsın
        // beyinden birebir üretilmeli — motora HİÇ gidilmeden (0 maliyet, bağımsız).
        let tmp = std::env::temp_dir().join("mimar_distill_test.mimar");
        let _ = std::fs::remove_file(&tmp);
        let mut b = Brain::open(Some(tmp.to_str().unwrap())).unwrap();
        b.teach("iade nasıl yapılır", "14 gün içinde kargoyla iade edebilirsiniz");
        b.teach("kargo kaç günde gelir", "2-3 iş günü");
        b.teach("çalışma saatleri", "hafta içi 09:00-18:00");
        // Ulaşılamaz motor: çağrılırsa test anlamını yitirir (hit → motor çağrılMAZ).
        let dead = Engine::ollama("yok-model"); // 11434 kapalıysa bile çağrılmaz
        let cases = vec![
            ("iade nasıl yapılır".to_string(), "14 gün içinde kargoyla iade edebilirsiniz".to_string()),
            ("kargo kaç günde gelir".to_string(), "2-3 iş günü".to_string()),
            ("çalışma saatleri".to_string(), "hafta içi 09:00-18:00".to_string()),
        ];
        let r = fidelity(&mut b, &dead, &cases, 0.9).unwrap();
        assert_eq!(r.oran, 1.0, "yakalanan davranış birebir üretilmeli (modelden bağımsız)");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn matches_exact_and_semantic() {
        assert_eq!(matches("evet", "evet", 0.9).0, true);
        assert_eq!(matches("  evet ", "evet", 0.9).0, true);   // trim
        assert_eq!(matches("tamamen alakasız bir cevap", "evet", 0.9).0, false);
    }
}

/// Beyin OLMADAN aynı motorun sadakati — karşılaştırma tabanı (baseline).
/// B+beyin ile B-yalnız arasındaki FARK = damıtılan davranışın taşınma miktarı.
pub fn raw_fidelity(engine: &Engine, system: &str, cases: &[(String, String)], thresh: f64)
    -> Result<FidReport, String> {
    let mut hit = 0;
    let mut detay = Vec::new();
    for (inp, expected) in cases {
        let out = engine.complete(system, inp)?;
        let (ok, s) = matches(&out, expected, thresh);
        if ok { hit += 1; }
        detay.push((inp.clone(), ok, s, out));
    }
    Ok(FidReport {
        oran: if cases.is_empty() { 0.0 } else { hit as f64 / cases.len() as f64 },
        uyan: hit, toplam: cases.len(), detay,
    })
}
