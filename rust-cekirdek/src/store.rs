//! Beyin store'u: .mimar öz formatını OKU + YAZ; teach/ask/öğren/guard.
//! Python referansıyla aynı format ve aynı eşikler → dosyalar birebir uyumlu.

use crate::embed;
use crate::embedder::{make_embedder, Embedder};
#[cfg(feature = "net")]
use crate::engine::Engine;
use crate::hash::{hmac_sha256, sha256, hex};
use crate::hnsw::Hnsw;
use crate::json::{self, J};
use crate::ml::LogReg;
use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};

const INDEX_THRESHOLD: usize = 64;

const SIGNED_FIELDS: [&str; 7] =
    ["role", "rules", "tools", "knowledge", "cache", "evals", "feature"];

pub const MAGIC: &[u8; 4] = b"MIMR";
pub const FORMAT_VERSION: u16 = 1;
pub const REUSE_SIM: f64 = 0.55;
pub const SAME_SIM: f64 = 0.78;
pub const LEARN_THRESHOLD: f64 = 2.0;
/// En yakın iki aday bu kadar yakınsa VE farklı cevaplıysa → belirsiz, çekimser.
/// ("ağırlıklar çok yakın" yanlış-pozitif riskine karşı marj güvenliği.)
pub const AMBIGUITY_MARGIN: f64 = 0.07;
/// Kalibrasyon eşiği TAVANI: eşik bunu aşmaz → recall ölmesin. "Benzer-ama-farklı-
/// cevap" yakın çiftlerini zaten MARJ-çekimser (sorgu anında) yakalar; kalibrasyonun
/// işi UZAK/alakasızları elemek, her şeyi bloklamak değil.
pub const CALIB_CEILING: f64 = 0.85;

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()
}

/// Guard karşılaştırması için normalizasyon: küçük harf + alt/üst-simge rakamları
/// ASCII'ye indir (örn. H₂O → h2o). Böylece model birim varyantı yazsa da yakalanır.
fn gnorm(s: &str) -> String {
    let mut out = String::new();
    for ch in s.chars() {
        let c = match ch {
            '\u{2080}'..='\u{2089}' =>
                char::from_u32('0' as u32 + (ch as u32 - 0x2080)).unwrap(),
            '⁰' => '0', '¹' => '1', '²' => '2', '³' => '3', '⁴' => '4',
            '⁵' => '5', '⁶' => '6', '⁷' => '7', '⁸' => '8', '⁹' => '9',
            _ => ch,
        };
        for l in c.to_lowercase() { out.push(l); }
    }
    out
}

fn jarr_f64(j: &J) -> Vec<f64> {
    if let J::Arr(a) = j { a.iter().filter_map(|x| x.as_f64()).collect() } else { Vec::new() }
}

/// Vektör oku — hem dense dizi [..] hem SEYREK nesne {n,i,v} formatını anlar
/// (geriye-uyumlu: eski dosyalar dense, yeni dosyalar seyrek).
fn read_vec(j: &J) -> Vec<f64> {
    match j {
        J::Arr(_) => jarr_f64(j),
        J::Obj(_) => {
            let n = j.get("n").and_then(|v| v.as_f64()).unwrap_or(embed::DIM as f64) as usize;
            let mut out = vec![0.0; n];
            if let (Some(J::Arr(idx)), Some(J::Arr(val))) = (j.get("i"), j.get("v")) {
                for (a, b) in idx.iter().zip(val.iter()) {
                    if let (Some(ix), Some(vv)) = (a.as_f64(), b.as_f64()) {
                        let ix = ix as usize;
                        if ix < out.len() { out[ix] = vv; }
                    }
                }
            }
            out
        }
        _ => Vec::new(),
    }
}

/// Varsayılan değerli alanı dosyaya YAZMA (okurken zaten varsayılan atanıyor).
/// 20k kayıtta alan-başı ~15 bayt × birçok alan = MB'larca kazanç.
fn is_default(k: &str, v: &J) -> bool {
    match k {
        "collection" => v.as_str() == Some("kesin"),
        "tool" => v.as_str() == Some(""),
        "volatility" => v.as_str() == Some("statik"),
        "scope" => v.as_str() == Some("global"),
        "slots" => matches!(v, J::Arr(a) if a.is_empty()),
        "ttl_sec" => v.as_f64() == Some(-1.0),
        "n" | "repeats" => v.as_f64() == Some(1.0),
        "hits" => v.as_f64() == Some(0.0),
        "blocked" | "learned" => matches!(v, J::Bool(false)),
        "verified" => matches!(v, J::Bool(true)),
        _ => false,
    }
}
fn strip_defaults(e: &mut J) {
    if let J::Obj(m) = e { m.retain(|(k, v)| !is_default(k, v)); }
}

/// Vektör yaz — yarıdan fazlası sıfırsa SEYREK {n,i,v} (hash-embedder ~%84 sıfır
/// → ~6x küçük), aksi halde dense dizi.
fn write_vec(v: &[f64]) -> J {
    let nz: Vec<usize> = v.iter().enumerate().filter(|(_, x)| **x != 0.0).map(|(i, _)| i).collect();
    if nz.len() * 2 < v.len() {
        J::Obj(vec![
            ("n".into(), J::n(v.len() as f64)),
            ("i".into(), J::Arr(nz.iter().map(|i| J::n(*i as f64)).collect())),
            ("v".into(), J::Arr(nz.iter().map(|i| J::n(v[*i])).collect())),
        ])
    } else {
        J::Arr(v.iter().map(|x| J::n(*x)).collect())
    }
}

fn new_data(feature: &str) -> J {
    J::Obj(vec![
        ("mimar".into(), J::n(FORMAT_VERSION as f64)),
        ("created".into(), J::n(now())),
        ("role".into(), J::s("Yardımcı bir asistan.")),
        ("config".into(), J::Obj(vec![])),
        ("rules".into(), J::Arr(vec![])),
        ("tools".into(), J::Arr(vec![])),
        ("knowledge".into(), J::Arr(vec![])),
        ("memory".into(), J::Arr(vec![])),
        ("evals".into(), J::Arr(vec![])),
        ("cache".into(), J::Obj(vec![])),
        ("stats".into(), J::Obj(vec![
            ("op_ms".into(), J::Obj(vec![])), ("runs".into(), J::n(0.0))])),
        ("log".into(), J::Arr(vec![])),
        ("feature".into(), J::s(feature)),
    ])
}

pub struct Brain {
    pub path: Option<String>,
    pub raw: J,
    pub ml: LogReg,         // motora gömülü öğrenen sınıflandırıcı (kalıcı)
    idx: Option<Hnsw>,      // büyük önbellekte anlamsal indeks (HNSW, tembel)
    embedder: Box<dyn Embedder>,  // takılabilir (varsayılan hashing; opt-in nöral)
}

pub struct Hit {
    pub key: String,
    pub sim: f64,
    pub answer: String,
    pub verified: bool,
    pub blocked: bool,
    pub learned: bool,
    pub collection: String,       // kesin/kural/arac/uretim/kisisel/eylem
    pub tool: String,             // arac için bağlı fonksiyon adı
    pub slots: Vec<String>,       // kural/arac için slot adları
    pub ttl_sec: f64,             // -1 = yok(statik), 0 = asla cache, >0 = tazelik
    pub ts: f64,                  // son yazılma zamanı (tazelik için)
    pub scope: String,            // "global" veya "user:<id>"
}

/// Bir yazma işleminin dürüst sonucu — "öğretildi" yalanını önler.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum PutOutcome { Created, Updated, Unchanged }

impl PutOutcome {
    pub fn label(&self) -> &'static str {
        match self {
            PutOutcome::Created => "yeni niyet eklendi",
            PutOutcome::Updated => "mevcut niyet güncellendi (üstüne yazıldı)",
            PutOutcome::Unchanged => "değişmedi (korunan taught niyet; güncellemek için 'forget' sonra 'teach')",
        }
    }
}

/// Niyet meta verisi (opsiyonel; teach için hepsi varsayılan = statik kesin gerçek).
#[derive(Default, Clone)]
pub struct Meta {
    pub collection: Option<String>,
    pub tool: Option<String>,
    pub slots: Vec<String>,
    pub ttl_sec: Option<f64>,
    pub volatility: Option<String>,
    pub scope: Option<String>,   // "global" (vars.) veya "user:<id>"
}

/// PII redaksiyon (zero-dep): e-posta/telefon/TCKN/kart maskele. Paylaşılan
/// (global) beyne yazmadan ÖNCE host bunu çağırmalı — PII paylaşılana girmesin.
fn pii_label(digit_str: &str) -> &'static str {
    match digit_str.len() {
        15 | 16 => "[kart]",
        11 => if digit_str.starts_with('0') { "[telefon]" } else { "[TC]" },  // TCKN 0 ile başlamaz
        _ => "[telefon]",
    }
}

pub fn redact(s: &str) -> String {
    let mut out = String::new();
    let b: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        // e-posta: harf/nokta/@ içeren token
        if c.is_alphanumeric() || c == '.' || c == '_' || c == '+' || c == '-' || c == '@' {
            let start = i;
            while i < b.len() && (b[i].is_alphanumeric() || "._+-@".contains(b[i])) { i += 1; }
            let tok: String = b[start..i].iter().collect();
            let digits: String = tok.chars().filter(|c| c.is_ascii_digit()).collect();
            if tok.contains('@') && tok.contains('.') {
                out.push_str("[e-posta]");
            } else if digits.len() >= 10 {
                out.push_str(pii_label(&digits));
            } else {
                out.push_str(&tok);
            }
        } else if c.is_ascii_digit() || c == ' ' || c == '(' || c == ')' {
            // ayraçlı rakam dizisi (ör. "0555 123 45 67")
            let start = i;
            while i < b.len() && (b[i].is_ascii_digit() || " ()-".contains(b[i])) { i += 1; }
            let run: String = b[start..i].iter().collect();
            let digits: String = run.chars().filter(|c| c.is_ascii_digit()).collect();
            if digits.len() >= 10 {
                let lead: String = run.chars().take_while(|c| *c == ' ').collect();
                let trail: String = run.chars().rev().take_while(|c| *c == ' ').collect();
                out.push_str(&lead);
                out.push_str(pii_label(&digits));
                out.push_str(&trail);
            } else {
                out.push_str(&run);
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

/// recall/resolve'un tipli sonucu — mimar YÖNLENDİRİR, host ÇALIŞTIRIR.
#[derive(Debug, Clone, PartialEq)]
pub enum Directive {
    /// kesin gerçek: doğrudan kullan (bedava/anında).
    Fact { answer: String, sim: f64, learned: bool },
    /// kural: slotları çıkardık; host `handler` mantığını çalıştırır (ör. saat>=14 → olur).
    Rule { handler: String, query: String, slots: Vec<(String, String)>, sim: f64 },
    /// arac/eylem: host fonksiyonu/eylemi çalıştırır; değer ASLA cache'lenmez.
    Tool { tool: String, slots: Vec<(String, String)>, sim: f64, action: bool },
    /// tazelik doldu → yeniden çek/doğrula.
    Stale { collection: String, sim: f64 },
    /// hafızada yok / eşik altı / mod engelli → host/model taze cevaplar.
    Miss { sim: f64, reason: String },
}

impl Directive {
    /// Host'un kolay ayrıştırması için JSON. CLI ve HTTP servisi ikisi de kullanır.
    pub fn to_json(&self) -> J {
        fn slots_obj(s: &[(String, String)]) -> J {
            J::Obj(s.iter().map(|(k, v)| (k.clone(), J::s(v))).collect())
        }
        match self {
            Directive::Fact { answer, sim, learned } => J::Obj(vec![
                ("tip".into(), J::s("fact")), ("sim".into(), J::n(*sim)),
                ("learned".into(), J::Bool(*learned)), ("cevap".into(), J::s(answer))]),
            Directive::Rule { handler, query, slots, sim } => J::Obj(vec![
                ("tip".into(), J::s("rule")), ("sim".into(), J::n(*sim)),
                ("handler".into(), J::s(handler)), ("key".into(), J::s(query)),
                ("slots".into(), slots_obj(slots))]),
            Directive::Tool { tool, slots, sim, action } => J::Obj(vec![
                ("tip".into(), J::s("tool")), ("sim".into(), J::n(*sim)),
                ("tool".into(), J::s(tool)), ("action".into(), J::Bool(*action)),
                ("slots".into(), slots_obj(slots))]),
            Directive::Stale { collection, sim } => J::Obj(vec![
                ("tip".into(), J::s("stale")), ("sim".into(), J::n(*sim)),
                ("collection".into(), J::s(collection))]),
            Directive::Miss { sim, reason } => J::Obj(vec![
                ("tip".into(), J::s("miss")), ("sim".into(), J::n(*sim)),
                ("reason".into(), J::s(reason))]),
        }
    }
}

/// Çalışma-anı güven kolu (config.policy'den okunur; canlı ayarlanır).
pub struct Policy {
    pub mode: String,       // karma / sadece_yz / sadece_mimar
    pub esik: f64,          // karma'da min güven (sim) eşiği
    pub ogrenme: bool,      // yeni öğrenme açık mı
}

// ---- slot çıkarımı (bağımlılıksız; TR-dostu) ----
/// Saat: "16", "15.00", "14:00" → saat tamsayısı "16".
fn slot_time(q: &str) -> Option<String> {
    let b = q.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit() {
            let mut j = i;
            while j < b.len() && b[j].is_ascii_digit() { j += 1; }
            if let Ok(h) = q[i..j].parse::<u32>() {
                if h <= 23 && (j - i) <= 2 {
                    // ardından :MM / .MM gelebilir ama yalnız saati döndürüyoruz
                    return Some(h.to_string());
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }
    None
}

/// Sayı/kişi: ilk tamsayı.
fn slot_num(q: &str) -> Option<String> {
    let b = q.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit() {
            let mut j = i;
            while j < b.len() && b[j].is_ascii_digit() { j += 1; }
            return Some(q[i..j].to_string());
        }
        i += 1;
    }
    None
}

/// Tarih: "25 ekim" / "25.10" / "25/10" → ham eşleşen ifade (host yorumlar).
fn slot_date(q: &str) -> Option<String> {
    let aylar = ["ocak","şubat","subat","mart","nisan","mayıs","mayis","haziran",
                 "temmuz","ağustos","agustos","eylül","eylul","ekim","kasım","kasim","aralık","aralik"];
    let low = q.to_lowercase();
    // "DD ay"
    let toks: Vec<&str> = low.split(|c: char| !c.is_alphanumeric()).filter(|s| !s.is_empty()).collect();
    for w in 0..toks.len() {
        if toks[w].chars().all(|c| c.is_ascii_digit()) && toks[w].len() <= 2 {
            if w + 1 < toks.len() && aylar.contains(&toks[w + 1]) {
                return Some(format!("{} {}", toks[w], toks[w + 1]));
            }
        }
    }
    // "DD.MM" / "DD/MM"
    let b = low.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit() {
            let mut j = i;
            while j < b.len() && (b[j].is_ascii_digit() || b[j] == b'.' || b[j] == b'/') { j += 1; }
            let s = &low[i..j];
            if (s.contains('.') || s.contains('/')) && s.len() >= 4 {
                return Some(s.to_string());
            }
            i = j;
        } else { i += 1; }
    }
    None
}

/// Türkçe karakterleri ASCII'ye indir + küçük harf (eşleştirme için; İ/I dahil).
fn fold(s: &str) -> String {
    let mut o = String::new();
    for ch in s.chars() {
        let c = match ch {
            'ç' | 'Ç' => 'c', 'ş' | 'Ş' => 's', 'ğ' | 'Ğ' => 'g',
            'ü' | 'Ü' => 'u', 'ö' | 'Ö' => 'o', 'ı' | 'I' => 'i',
            'İ' => 'i', 'â' | 'Â' => 'a', 'î' => 'i', 'û' => 'u',
            _ => { for l in ch.to_lowercase() { o.push(l); } continue; }
        };
        o.push(c);
    }
    o
}

const ILLER: &[&str] = &[
    "adana","adıyaman","afyonkarahisar","ağrı","amasya","ankara","antalya","artvin",
    "aydın","balıkesir","bilecik","bingöl","bitlis","bolu","burdur","bursa","çanakkale",
    "çankırı","çorum","denizli","diyarbakır","edirne","elazığ","erzincan","erzurum",
    "eskişehir","gaziantep","giresun","gümüşhane","hakkari","hatay","ısparta","mersin",
    "istanbul","izmir","kars","kastamonu","kayseri","kırklareli","kırşehir","kocaeli",
    "konya","kütahya","malatya","manisa","kahramanmaraş","mardin","muğla","muş","nevşehir",
    "niğde","ordu","rize","sakarya","samsun","siirt","sinop","sivas","tekirdağ","tokat",
    "trabzon","tunceli","şanlıurfa","uşak","van","yozgat","zonguldak","aksaray","bayburt",
    "karaman","kırıkkale","batman","şırnak","bartın","ardahan","iğdır","yalova","karabük",
    "kilis","osmaniye","düzce",
];

/// Şehir: sorgudaki token'ları 81 il listesiyle (Türkçe-fold'lu) eşleştir.
fn slot_city(q: &str) -> Option<String> {
    let fq = fold(q);
    for il in ILLER {
        if fq.contains(&fold(il)) { return Some((*il).to_string()); }
    }
    None
}

fn extract_slots(query: &str, names: &[String]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for n in names {
        let v = match n.as_str() {
            "saat" => slot_time(query),
            "tarih" => slot_date(query),
            "sayi" | "kisi" | "kişi" => slot_num(query),
            "sehir" | "şehir" | "il" => slot_city(query),
            // Herhangi ÖZEL slot adı için genel çıkarım: "ad değer" / "ad=değer" / "ad: değer".
            _ => slot_generic(query, n),
        };
        if let Some(v) = v { out.push((n.clone(), v)); }
    }
    out
}

/// Genel slot çıkarımı: sorguda slot ADINI bul, değerini al. Üç biçim:
/// "ad=değer", "ad: değer", "ad değer" (sonraki token). Dilden bağımsız (ad eşleşmesi).
fn slot_generic(query: &str, name: &str) -> Option<String> {
    let nl = name.to_lowercase();
    let toks: Vec<&str> = query.split_whitespace().collect();
    for (i, t) in toks.iter().enumerate() {
        // tek-token "ad=değer" / "ad:değer"
        for sep in ['=', ':'] {
            if let Some((k, v)) = t.split_once(sep) {
                if k.to_lowercase() == nl && !v.is_empty() {
                    return Some(v.trim_matches(|c| c == '=' || c == ':').to_string());
                }
            }
        }
        // "ad" ardından sonraki token değer
        let tnorm = t.trim_end_matches(['=', ':']).to_lowercase();
        if tnorm == nl {
            if let Some(nx) = toks.get(i + 1) {
                let val = nx.trim_start_matches(['=', ':']);
                if !val.is_empty() { return Some(val.to_string()); }
            }
        }
    }
    None
}

impl Brain {
    pub fn open(path: Option<&str>) -> Result<Brain, String> {
        let (path_s, raw) = match path {
            Some(p) if std::path::Path::new(p).exists() =>
                (Some(p.to_string()), read_container(p)?),
            Some(p) => (Some(p.to_string()), new_data("hash-256")),
            None => (None, new_data("hash-256")),
        };
        // Embedder'ı beynin `feature` kimliğinden kur (hash-256 ya da ollama:<model>).
        let feature = raw.get("feature").and_then(|v| v.as_str())
            .unwrap_or("hash-256").to_string();
        let embedder = make_embedder(&feature);
        let ml = match raw.get("ml") {
            Some(j) => LogReg::from_json(j),
            None => LogReg::new(embedder.dim(), 0.3),
        };
        // Büyük önbellekte LSH indeksini kur (küçükte lineer zaten hızlı+kesin).
        let mut idx = None;
        if let Some(J::Obj(cache)) = raw.get("cache") {
            if cache.len() >= INDEX_THRESHOLD {
                let mut l = Hnsw::new();
                for (k, c) in cache {
                    if let Some(v) = c.get("vec") {
                        let vv = read_vec(v);
                        if !vv.is_empty() { l.add(k, &vv); }
                    }
                }
                idx = Some(l);
            }
        }
        Ok(Brain { path: path_s, raw, ml, idx, embedder })
    }

    /// Embedder'ı değiştir (opt-in nöral) + TÜM vektörleri yeniden hesapla + ML/indeks
    /// yeni boyuta sıfırla + feature güncelle. Uzay tutarlılığı korunur.
    pub fn set_embedder(&mut self, embedder: Box<dyn Embedder>) -> usize {
        self.embedder = embedder;
        let id = self.embedder.id();
        self.raw.set("feature", J::s(&id));
        // niyet vektörlerini yeni embedder'la yeniden hesapla
        let keys_queries: Vec<(String, String)> = match self.raw.get("cache") {
            Some(J::Obj(m)) => m.iter().map(|(k, c)|
                (k.clone(), c.get("query").and_then(|q| q.as_str()).unwrap_or("").to_string())
            ).collect(),
            _ => Vec::new(),
        };
        let mut n = 0;
        for (k, q) in &keys_queries {
            let v = self.embedder.embed(q);
            if let Some(cache) = self.raw.get_mut("cache").and_then(|c| c.obj_entries_mut()) {
                if let Some(slot) = cache.iter_mut().find(|(kk, _)| kk == k) {
                    slot.1.set("vec", write_vec(&v));
                    n += 1;
                }
            }
        }
        // ML ve indeks yeni boyuta göre sıfırlanır (eski ağırlıklar farklı boyuttaydı)
        self.ml = LogReg::new(self.embedder.dim(), 0.3);
        self.idx = None;
        n
    }

    pub fn embedder_id(&self) -> String { self.embedder.id() }

    /// CONFORMAL KALİBRASYON: güven eşiğini sihirli sayı yerine beynin KENDİ
    /// verisinden türet. Farklı-cevaplı niyet çiftlerinin benzerlik dağılımına
    /// bakıp eşiği onların ~%98'inin ÜSTÜNE koyar → alakasız/farklı niyet servis
    /// edilemez (garantili düşük yanlış-pozitif). En zor birkaç karışık çifti marj
    /// güvenliği yakalar. Eşiği policy.esik'e yazar (resolve onu kullanır).
    pub fn calibrate(&mut self) -> (f64, usize, f64) {
        let ents = self.entries();
        let live: Vec<&Ent> = ents.iter()
            .filter(|e| !e.blocked && !e.vec.is_empty()).collect();
        let n = live.len();
        let mut cross: Vec<f64> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if live[i].answer == live[j].answer { continue; }  // aynı cevap = risk değil
                cross.push(embed::cosine(&live[i].vec, &live[j].vec));
            }
        }
        let bar = if cross.is_empty() {
            REUSE_SIM
        } else {
            cross.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // p90 (yığının üstü) + küçük pay; yakın-çiftlerin EN yükseğine değil.
            // Marj-çekimser zaten yakın-farklı çiftleri sorgu anında ayıklar, bu yüzden
            // eşiği tavana (CALIB_CEILING) sınırlıyoruz → recall-öldüren 0.95'ler olmaz.
            let p90 = cross[((cross.len() as f64 * 0.90) as usize).min(cross.len() - 1)];
            (p90 + 0.02).min(CALIB_CEILING).max(REUSE_SIM)
        };
        self.set_policy(None, Some(bar), None);   // policy.esik = kalibre eşik
        let maxc = cross.iter().cloned().fold(0.0f64, f64::max);
        (bar, n, maxc)
    }

    pub fn save(&mut self) -> Result<(), String> {
        // Öğrenilen model ağırlıklarını beyin dosyasına göm (kalıcı ML).
        let mlj = self.ml.to_json();
        self.raw.set("ml", mlj);
        if let Some(p) = &self.path { write_container(p, &self.raw)?; }
        Ok(())
    }

    /// Hem anlamsal (cosine) hem öğrenen (lojistik) yönlendirmeyi göster.
    pub fn route(&self, query: &str) -> (Option<Hit>, Option<(String, f64)>) {
        let qv = self.embedder.embed(query);
        let ml = self.ml.predict(&qv)
            .map(|(k, p)| (self.query_of(&k).unwrap_or(k), p));
        (self.best_intent(query), ml)
    }

    /// Konsolidasyon: benzer niyetleri birleştir, uzun-kuyruğu buda, sınırla.
    /// Öğretilmiş/öğrenilmiş/engelli KORUNUR → davranış bozulmaz, dosya şişmez.
    /// Döndürür: (önce, birleşme_sonrası, sonra, budanan).
    pub fn consolidate(&mut self, max_intents: usize) -> (usize, usize, usize, usize) {
        let ents = self.entries();
        let before = ents.len();
        let (blocked, active): (Vec<Ent>, Vec<Ent>) =
            ents.into_iter().partition(|e| e.blocked);

        // Açgözlü kümeleme (cosine >= SAME_SIM)
        let mut used = vec![false; active.len()];
        let mut merged: Vec<Ent> = Vec::new();
        for i in 0..active.len() {
            if used[i] { continue; }
            used[i] = true;
            let mut cluster = vec![i];
            for j in (i + 1)..active.len() {
                if used[j] { continue; }
                if embed::cosine(&active[i].vec, &active[j].vec) >= SAME_SIM {
                    used[j] = true;
                    cluster.push(j);
                }
            }
            if cluster.len() == 1 {
                merged.push(active[i].clone());
            } else {
                // kazanan: (reward, repeats)
                let win = *cluster.iter().max_by(|&&a, &&b| {
                    (active[a].reward, active[a].repeats)
                        .partial_cmp(&(active[b].reward, active[b].repeats)).unwrap()
                }).unwrap();
                let mut m = active[win].clone();
                m.repeats = cluster.iter().map(|&x| active[x].repeats).sum();
                m.hits = cluster.iter().map(|&x| active[x].hits).sum();
                m.reward = cluster.iter().map(|&x| active[x].reward).fold(f64::MIN, f64::max);
                m.learned = cluster.iter().any(|&x| active[x].learned);
                // ağırlıklı ortalama vektör (boyut = mevcut vektörlerin boyutu)
                let total_n: f64 = cluster.iter().map(|&x| active[x].n).sum();
                let dim = m.vec.len();
                let mut v = vec![0.0; dim];
                for &x in &cluster {
                    for k in 0..dim.min(active[x].vec.len()) {
                        v[k] += active[x].vec[k] * active[x].n;
                    }
                }
                if total_n > 0.0 { for k in 0..v.len() { v[k] /= total_n; } }
                embed::normalize(&mut v);
                m.vec = v;
                merged.push(m);
            }
        }
        let after_merge = merged.len() + blocked.len();

        // Budama (kapasite) — korunanlar dokunulmaz
        let mut evicted = 0;
        if max_intents > 0 && merged.len() + blocked.len() > max_intents {
            let (mut protected, mut others): (Vec<Ent>, Vec<Ent>) =
                merged.into_iter().partition(|e| e.learned || e.source == "taught");
            others.sort_by(|a, b| (b.reward, b.hits, b.repeats, b.ts)
                .partial_cmp(&(a.reward, a.hits, a.repeats, a.ts)).unwrap());
            let room = max_intents.saturating_sub(protected.len() + blocked.len());
            evicted = others.len().saturating_sub(room);
            others.truncate(room);
            protected.append(&mut others);
            merged = protected;
        }

        // Yeniden kur
        let mut all = merged;
        all.extend(blocked);
        let obj: Vec<(String, J)> = all.iter().map(|e| (e.key.clone(), e.to_json())).collect();
        self.raw.set("cache", J::Obj(obj));
        self.idx = None;    // indeks bayatladı → sonraki açılışta yeniden kurulur
        let after = all.len();
        (before, after_merge, after, evicted)
    }

    fn entries(&self) -> Vec<Ent> {
        let mut out = Vec::new();
        if let Some(J::Obj(cache)) = self.raw.get("cache") {
            for (k, c) in cache {
                out.push(Ent {
                    key: k.clone(),
                    vec: match c.get("vec") { Some(v) => read_vec(v), None => Vec::new() },
                    n: c.get("n").and_then(|v| v.as_f64()).unwrap_or(1.0),
                    repeats: c.get("repeats").and_then(|v| v.as_f64()).unwrap_or(1.0),
                    hits: c.get("hits").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    reward: c.get("reward").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    ts: c.get("ts").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    answer: c.get("answer").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    query: c.get("query").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    verified: c.get("verified").map_or(true, |v| v.as_bool()),
                    learned: c.get("learned").map_or(false, |v| v.as_bool()),
                    blocked: c.get("blocked").map_or(false, |v| v.as_bool()),
                    source: c.get("source").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    collection: c.get("collection").and_then(|v| v.as_str()).unwrap_or("kesin").to_string(),
                    tool: c.get("tool").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    slots: match c.get("slots") { Some(J::Arr(a)) =>
                        a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect(), _ => Vec::new() },
                    ttl_sec: c.get("ttl_sec").and_then(|v| v.as_f64()).unwrap_or(-1.0),
                    volatility: c.get("volatility").and_then(|v| v.as_str()).unwrap_or("statik").to_string(),
                    scope: c.get("scope").and_then(|v| v.as_str()).unwrap_or("global").to_string(),
                });
            }
        }
        out
    }

    /// Batch eğitim (ANALYZE benzeri): beynin TÜM niyetleri üzerinde ML'i N
    /// epoch eğit. Online tek-adım zayıf kalır; bu geçiş sınıflandırıcıyı
    /// gerçekten ayırt eder hale getirir. Döndürür: örnek sayısı.
    pub fn fit(&mut self, epochs: usize) -> usize {
        let samples: Vec<(Vec<f64>, String)> = match self.raw.get("cache") {
            Some(J::Obj(cache)) => cache.iter().filter_map(|(k, c)| {
                c.get("query").and_then(|q| q.as_str())
                    .map(|q| (self.embedder.embed(q), k.clone()))
            }).collect(),
            _ => Vec::new(),
        };
        for _ in 0..epochs {
            for (x, lbl) in &samples {
                self.ml.train(x, lbl);
            }
        }
        samples.len()
    }

    // ---- imzalama (dağıtım güveni) ----
    pub fn content_hash(&self) -> String {
        let core: Vec<(String, J)> = SIGNED_FIELDS.iter()
            .filter_map(|f| self.raw.get(f).map(|v| (f.to_string(), v.clone())))
            .collect();
        hex(&sha256(json::write(&J::Obj(core)).as_bytes()))
    }
    pub fn sign(&mut self, key: &str, author: &str) -> String {
        let ch = self.content_hash();
        let sig = hmac_sha256(key.as_bytes(), ch.as_bytes());
        self.raw.set("signature", J::Obj(vec![
            ("alg".into(), J::s("hmac-sha256")), ("author".into(), J::s(author)),
            ("content_hash".into(), J::s(&ch)), ("sig".into(), J::s(&sig)),
            ("ts".into(), J::n(now()))]));
        format!("imzalandı (author={}, hash={}…)", author, &ch[..16])
    }
    pub fn verify(&self, key: &str) -> (bool, String) {
        let s = match self.raw.get("signature") { Some(s) => s, None => return (false, "imza yok".into()) };
        let stored = s.get("content_hash").and_then(|v| v.as_str()).unwrap_or("");
        let ch = self.content_hash();
        if ch != stored {
            return (false, "içerik değişmiş (kurcalama tespit edildi)".into());
        }
        let sig = s.get("sig").and_then(|v| v.as_str()).unwrap_or("");
        let expect = hmac_sha256(key.as_bytes(), ch.as_bytes());
        if expect == sig {
            (true, format!("geçerli (author={})", s.get("author").and_then(|v| v.as_str()).unwrap_or("")))
        } else {
            (false, "imza/anahtar uyuşmuyor".into())
        }
    }

    fn query_of(&self, key: &str) -> Option<String> {
        if let Some(J::Obj(cache)) = self.raw.get("cache") {
            for (k, c) in cache {
                if k == key {
                    return c.get("query").and_then(|q| q.as_str()).map(|s| s.to_string());
                }
            }
        }
        None
    }

    pub fn role(&self) -> String {
        self.raw.get("role").and_then(|v| v.as_str()).unwrap_or("").to_string()
    }
    pub fn set_role(&mut self, r: &str) { self.raw.set("role", J::s(r)); }

    pub fn feature(&self) -> String {
        self.raw.get("feature").and_then(|v| v.as_str()).unwrap_or("?").to_string()
    }

    pub fn cache_len(&self) -> usize {
        match self.raw.get("cache") { Some(J::Obj(m)) => m.len(), _ => 0 }
    }

    /// Native GUI için niyet satırları: (query, answer, source, learned, blocked).
    pub fn intent_rows(&self) -> Vec<(String, String, String, bool, bool)> {
        let mut out = Vec::new();
        if let Some(J::Obj(cache)) = self.raw.get("cache") {
            for (_k, c) in cache {
                out.push((
                    c.get("query").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    c.get("answer").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    c.get("source").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    c.get("learned").map_or(false, |v| v.as_bool()),
                    c.get("blocked").map_or(false, |v| v.as_bool()),
                ));
            }
        }
        out
    }

    /// GUI/API için niyet listesi (query, answer, source, learned, blocked).
    pub fn intents_json(&self) -> J {
        let mut out = Vec::new();
        if let Some(J::Obj(cache)) = self.raw.get("cache") {
            for (_k, c) in cache {
                out.push(J::Obj(vec![
                    ("query".into(), J::s(c.get("query").and_then(|v| v.as_str()).unwrap_or(""))),
                    ("answer".into(), J::s(c.get("answer").and_then(|v| v.as_str()).unwrap_or(""))),
                    ("source".into(), J::s(c.get("source").and_then(|v| v.as_str()).unwrap_or(""))),
                    ("learned".into(), J::Bool(c.get("learned").map_or(false, |v| v.as_bool()))),
                    ("blocked".into(), J::Bool(c.get("blocked").map_or(false, |v| v.as_bool()))),
                ]));
            }
        }
        J::Arr(out)
    }

    /// DONDUR: beyni gömülü int8 pakete nicemle (MCU/WASM/4GB). Engelli/boş-vektör
    /// atlanır; sadece kesin-gerçek cevaplar alınır (kural/araç host-yürütmesi gömülüde yok).
    pub fn freeze(&self) -> crate::quant::QuantPack {
        let mut pack = crate::quant::QuantPack::new(self.embedder.dim());
        if let Some(J::Obj(cache)) = self.raw.get("cache") {
            for (_k, c) in cache {
                if c.get("blocked").map_or(false, |v| v.as_bool()) { continue; }
                let coll = c.get("collection").and_then(|v| v.as_str()).unwrap_or("kesin");
                if coll != "kesin" { continue; }   // gömülüde sadece doğrudan cevap
                let vec = match c.get("vec") { Some(v) => read_vec(v), None => continue };
                if vec.is_empty() { continue; }
                let ans = c.get("answer").and_then(|a| a.as_str()).unwrap_or("");
                pack.add(&vec, ans);
            }
        }
        pack
    }

    // ---- guard'lar (çıktıda yasak desenler) ----
    pub fn guard_patterns(&self) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(J::Arr(rules)) = self.raw.get("rules") {
            for r in rules {
                if r.get("kind").and_then(|k| k.as_str()) == Some("guard") {
                    if let Some(f) = r.get("forbid").and_then(|f| f.as_str()) {
                        out.push(f.to_string());
                    }
                }
            }
        }
        out
    }
    pub fn remove_guard(&mut self, text: &str) {
        if let Some(rules) = self.raw.get_mut("rules").and_then(|r| r.obj_entries_mut_arr()) {
            rules.retain(|r| !(r.get("kind").and_then(|k| k.as_str()) == Some("guard")
                && r.get("forbid").and_then(|f| f.as_str()) == Some(text)));
        }
    }

    // ---- GUI yönetimi: niyet düzenle / sil / engelle ----
    pub fn delete_intent(&mut self, query: &str) -> bool {
        let k = embed::key(query);
        let mut removed = false;
        if let Some(cache) = self.raw.get_mut("cache").and_then(|c| c.obj_entries_mut()) {
            let before = cache.len();
            cache.retain(|(kk, _)| *kk != k);
            removed = cache.len() != before;
        }
        if removed { self.idx = None; }
        removed
    }
    pub fn edit_answer(&mut self, query: &str, new_answer: &str) -> bool {
        let k = embed::key(query);
        if let Some(cache) = self.raw.get_mut("cache").and_then(|c| c.obj_entries_mut()) {
            if let Some(slot) = cache.iter_mut().find(|(kk, _)| *kk == k) {
                slot.1.set("answer", J::s(new_answer));
                return true;
            }
        }
        false
    }
    pub fn set_blocked(&mut self, query: &str, blocked: bool) -> bool {
        let k = embed::key(query);
        if let Some(cache) = self.raw.get_mut("cache").and_then(|c| c.obj_entries_mut()) {
            if let Some(slot) = cache.iter_mut().find(|(kk, _)| *kk == k) {
                slot.1.set("blocked", J::Bool(blocked));
                if blocked { slot.1.set("learned", J::Bool(false)); }
                return true;
            }
        }
        false
    }

    // ---- GUI yönetimi: hafıza ----
    pub fn memory_rows(&self) -> Vec<String> {
        match self.raw.get("memory") {
            Some(J::Arr(a)) => a.iter()
                .filter_map(|m| m.get("text").and_then(|t| t.as_str()).map(|s| s.to_string()))
                .collect(),
            _ => Vec::new(),
        }
    }
    pub fn add_memory(&mut self, text: &str, kind: &str) {
        let id = match self.raw.get("memory") {
            Some(J::Arr(a)) => a.iter().filter_map(|m| m.get("id").and_then(|i| i.as_f64()))
                .fold(0.0, f64::max) + 1.0,
            _ => 1.0,
        };
        if let Some(mem) = self.raw.get_mut("memory").and_then(|m| m.obj_entries_mut_arr()) {
            mem.push(J::Obj(vec![
                ("id".into(), J::n(id)), ("kind".into(), J::s(kind)),
                ("text".into(), J::s(text)), ("weight".into(), J::n(1.0)),
                ("created".into(), J::n(now())), ("used".into(), J::n(now()))]));
        }
    }
    pub fn clear_memory(&mut self) {
        self.raw.set("memory", J::Arr(vec![]));
    }

    pub fn add_guard(&mut self, text: &str) {
        let id = self.next_rule_id();
        if let Some(rules) = self.raw.get_mut("rules").and_then(|r| r.obj_entries_mut_arr()) {
            rules.push(J::Obj(vec![
                ("id".into(), J::n(id)), ("kind".into(), J::s("guard")),
                ("text".into(), J::s(text)), ("trigger".into(), J::Null),
                ("response".into(), J::Null), ("forbid".into(), J::s(text)),
                ("priority".into(), J::n(100.0))]));
        }
    }
    fn next_rule_id(&self) -> f64 {
        if let Some(J::Arr(rules)) = self.raw.get("rules") {
            rules.iter().filter_map(|r| r.get("id").and_then(|i| i.as_f64()))
                .fold(0.0, f64::max) + 1.0
        } else { 1.0 }
    }
    pub fn check_guard(&self, answer: &str) -> Option<String> {
        let low = gnorm(answer);
        self.guard_patterns().into_iter().find(|p| {
            let pn = gnorm(p);
            !pn.is_empty() && low.contains(&pn)
        })
    }

    /// Geri bildirim: kötü → niyeti engelle (reuse artık servis etmez).
    pub fn feedback(&mut self, query: &str, good: bool) -> String {
        let best = self.best_intent(query);
        let key = match &best {
            Some(h) if h.sim >= REUSE_SIM => h.key.clone(),
            _ => return "eşleşen niyet yok".into(),
        };
        if let Some(cache) = self.raw.get_mut("cache").and_then(|c| c.obj_entries_mut()) {
            if let Some(slot) = cache.iter_mut().find(|(k, _)| *k == key) {
                let r = slot.1.get("reward").and_then(|v| v.as_f64()).unwrap_or(0.0)
                    + if good { 1.0 } else { -1.0 };
                slot.1.set("reward", J::n(r));
                if !good {
                    slot.1.set("blocked", J::Bool(true));
                    slot.1.set("learned", J::Bool(false));
                    return "negatif → niyet engellendi".into();
                }
                return format!("pozitif → ödül {}", r);
            }
        }
        "niyet bulunamadı".into()
    }

    /// EXPLAIN: bu sorgu için hangi yol seçilir (incelenebilirlik).
    pub fn explain(&self, query: &str, risk_high: bool) -> String {
        let best = self.best_intent(query);
        let sim = best.as_ref().map_or(0.0, |b| b.sim);
        if risk_high {
            return format!("PLAN «{}»\n  1. MODEL (risk=high → kestirme yok, güçlü model)\n  \
                            2. VERIFY (guard)\n  3. COMMIT\n  (en yakın niyet sim={:.3})",
                           query, sim);
        }
        if let Some(h) = &best {
            if h.sim >= REUSE_SIM && h.verified && !h.blocked {
                let op = if h.learned { "RULE" } else { "CACHE_GET" };
                return format!("PLAN «{}»\n  1. {} (derlenmiş, sim={:.3}, ~0ms, $0)\n  \
                                2. VERIFY (guard)\n  → modeli ATLA", query, op, h.sim);
            }
        }
        format!("PLAN «{}»\n  1. RECALL\n  2. MODEL (sim={:.3} < {} → model gerek)\n  \
                 3. VERIFY (guard)\n  4. COMMIT (öğren)", query, sim, REUSE_SIM)
    }

    pub fn system_prompt(&self) -> String {
        let mut s = self.role();
        let guards = self.guard_patterns();
        if !guards.is_empty() {
            s.push_str("\n\nKesin yasaklar (bunları ASLA söyleme): ");
            s.push_str(&guards.join(", "));
        }
        s
    }

    // ---- anlamsal en-yakın niyet ----
    pub fn best_intent(&self, query: &str) -> Option<Hit> {
        self.best_intent_scoped(query, None)
    }

    /// Scope-filtreli en-yakın niyet. `allowed=None` → tüm scope'lar (iç kullanım/admin);
    /// `Some(u)` → yalnız "global" + u (başka kullanıcının verisi GÖRÜNMEZ = gizlilik).
    pub fn best_intent_scoped(&self, query: &str, allowed: Option<&str>) -> Option<Hit> {
        let qv = self.embedder.embed(query);
        let cache = match self.raw.get("cache") { Some(J::Obj(m)) => m, _ => return None };
        let cand: Option<HashSet<String>> = self.idx.as_ref().map(|ix| {
            let mut c = ix.candidates(&qv);
            c.insert(embed::key(query));
            c
        });
        let mut best: Option<Hit> = None;
        for (k, c) in cache {
            if let Some(set) = &cand {
                if !set.contains(k) { continue; }
            }
            let sc = c.get("scope").and_then(|v| v.as_str()).unwrap_or("global");
            if let Some(u) = allowed {              // scope filtresi (gizlilik)
                if sc != "global" && sc != u { continue; }
            }
            let vec = match c.get("vec") { Some(v) => read_vec(v), None => continue };
            if vec.is_empty() { continue; }
            let sim = embed::cosine(&qv, &vec);
            if best.as_ref().map_or(true, |b| sim > b.sim) {
                best = Some(Hit {
                    key: k.clone(), sim,
                    answer: c.get("answer").and_then(|a| a.as_str()).unwrap_or("").to_string(),
                    verified: c.get("verified").map_or(true, |v| v.as_bool()),
                    blocked: c.get("blocked").map_or(false, |v| v.as_bool()),
                    learned: c.get("learned").map_or(false, |v| v.as_bool()),
                    collection: c.get("collection").and_then(|v| v.as_str()).unwrap_or("kesin").to_string(),
                    tool: c.get("tool").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    slots: match c.get("slots") { Some(J::Arr(a)) =>
                        a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect(), _ => Vec::new() },
                    ttl_sec: c.get("ttl_sec").and_then(|v| v.as_f64()).unwrap_or(-1.0),
                    ts: c.get("ts").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    scope: sc.to_string(),
                });
            }
        }
        best
    }

    fn put_intent(&mut self, query: &str, answer: &str, source: &str,
                  verified: bool, learned: bool, reward: f64, meta: &Meta) -> PutOutcome {
        let k = embed::key(query);
        let qv = self.embedder.embed(query);
        // ogrenme=kapalı → ML eğitimini atla (panik kolu + toplu yükleme/ölçek).
        // resolve zaten cosine+LSH kullanır; ML yalnız route/öneri içindir.
        if self.policy().ogrenme { self.ml.train(&qv, &k); }
        let vec = write_vec(&qv);
        let collection = meta.collection.clone().unwrap_or_else(|| "kesin".into());
        let ttl = meta.ttl_sec.unwrap_or(-1.0);
        let scope = meta.scope.clone().unwrap_or_else(|| "global".into());
        let slots_j = J::Arr(meta.slots.iter().map(|s| J::s(s)).collect());
        let mut entry = J::Obj(vec![
            ("query".into(), J::s(query)), ("answer".into(), J::s(answer)),
            ("vec".into(), vec), ("n".into(), J::n(1.0)),
            ("hits".into(), J::n(0.0)), ("repeats".into(), J::n(1.0)),
            ("verified".into(), J::Bool(verified)), ("reward".into(), J::n(reward)),
            ("blocked".into(), J::Bool(false)), ("learned".into(), J::Bool(learned)),
            ("source".into(), J::s(source)), ("ts".into(), J::n(now())),
            ("collection".into(), J::s(&collection)),
            ("tool".into(), J::s(meta.tool.as_deref().unwrap_or(""))),
            ("slots".into(), slots_j),
            ("ttl_sec".into(), J::n(ttl)),
            ("volatility".into(), J::s(meta.volatility.as_deref().unwrap_or("statik"))),
            ("scope".into(), J::s(&scope)),
        ]);
        strip_defaults(&mut entry);   // varsayılan alanları yazma (boyut)
        // İKİ farklı yazar vardır ve aynı muameleyi görmemeleri gerekir:
        //  • "taught"  = kullanıcının AÇIK öğretmesi → YETKİLİ (eski taught olsa bile ezer: upsert).
        //  • "model"   = ask/note otomatik öğrenmesi → taught ground-truth'u ASLA ezmez.
        let incoming_taught = source == "taught";
        let mut outcome = PutOutcome::Created;
        let mut inserted = false;
        if let Some(cache) = self.raw.get_mut("cache").and_then(|c| c.obj_entries_mut()) {
            if let Some(slot) = cache.iter_mut().find(|(kk, _)| *kk == k) {
                let is_taught = slot.1.get("source").and_then(|s| s.as_str()) == Some("taught");
                if incoming_taught {
                    // Yetkili üst-yazma: cevabı güncelle + taught/verified/learned'a TERFİ ettir
                    // + varsa engeli kaldır. (Önceki model kaydı da burada taught'a yükselir → P2.)
                    slot.1.set("answer", J::s(answer));
                    slot.1.set("source", J::s("taught"));
                    slot.1.set("verified", J::Bool(verified));
                    slot.1.set("learned", J::Bool(learned));
                    slot.1.set("reward", J::n(reward));
                    slot.1.set("blocked", J::Bool(false));
                    slot.1.set("ts", J::n(now()));
                    slot.1.set("collection", J::s(&collection));
                    slot.1.set("tool", J::s(meta.tool.as_deref().unwrap_or("")));
                    slot.1.set("slots", J::Arr(meta.slots.iter().map(|s| J::s(s)).collect()));
                    slot.1.set("ttl_sec", J::n(ttl));
                    slot.1.set("volatility", J::s(meta.volatility.as_deref().unwrap_or("statik")));
                    slot.1.set("scope", J::s(&scope));
                    outcome = PutOutcome::Updated;
                } else if !is_taught {
                    // Otomatik öğrenme yalnız model kaydını günceller (ground-truth'a dokunmaz).
                    slot.1.set("answer", J::s(answer));
                    let rep = slot.1.get("repeats").and_then(|r| r.as_f64()).unwrap_or(1.0) + 1.0;
                    slot.1.set("repeats", J::n(rep));
                    if rep >= LEARN_THRESHOLD { slot.1.set("learned", J::Bool(true)); }
                    outcome = PutOutcome::Updated;
                } else {
                    // gelen=model, mevcut=taught → korunur, hiçbir şey değişmez
                    outcome = PutOutcome::Unchanged;
                }
            } else {
                cache.push((k.clone(), entry));
                inserted = true;
            }
        }
        if inserted {
            if let Some(idx) = &mut self.idx {
                idx.add(&k, &qv);        // indeksi artımlı güncelle
            } else {
                self.ensure_index();     // eşik yeni geçildiyse tembel kur
            }
            outcome = PutOutcome::Created;
        }
        outcome
    }

    /// Önbellek indeks eşiğini geçtiyse ve indeks yoksa HNSW'yi kur (tembel).
    /// Eşik altındaysa lineer tarama zaten hızlı+kesin — indeks gereksiz.
    fn ensure_index(&mut self) {
        if self.idx.is_some() { return; }
        let n = match self.raw.get("cache") { Some(J::Obj(m)) => m.len(), _ => 0 };
        if n < INDEX_THRESHOLD { return; }
        let mut l = Hnsw::new();
        if let Some(J::Obj(cache)) = self.raw.get("cache") {
            for (k, c) in cache {
                if let Some(v) = c.get("vec") {
                    let vv = read_vec(v);
                    if !vv.is_empty() { l.add(k, &vv); }
                }
            }
        }
        self.idx = Some(l);
    }

    /// Açık öğretme = YETKİLİ (statik kesin gerçek). Aynı soru varsa üstüne yazar.
    pub fn teach(&mut self, prompt: &str, answer: &str) -> PutOutcome {
        self.put_intent(prompt, answer, "taught", true, true, 1.0, &Meta::default())
    }

    /// Vektör-tabanlı öğretme (KİP-AGNOSTİK): host medya/veri parmak izini (`qv`),
    /// deterministik anahtarını (`key`) ve cevabını verir — embedder ATLANIR.
    /// Her zaman YETKİLİ (taught, kesin). Diske kaydetmez (çağıran `save()` eder).
    pub fn teach_vec(&mut self, key: &str, qv: &[f64], answer: &str) {
        if self.policy().ogrenme { self.ml.train(qv, key); }
        let entry = J::Obj(vec![
            ("answer".into(), J::s(answer)),
            ("vec".into(), write_vec(qv)),
            ("verified".into(), J::Bool(true)),
            ("source".into(), J::s("taught")),
            ("collection".into(), J::s("kesin")),
            ("ts".into(), J::n(now())),
        ]);
        let mut inserted = false;
        if let Some(cache) = self.raw.get_mut("cache").and_then(|c| c.obj_entries_mut()) {
            if let Some(slot) = cache.iter_mut().find(|(kk, _)| kk == key) {
                slot.1.set("answer", J::s(answer));
                slot.1.set("vec", write_vec(qv));
                slot.1.set("verified", J::Bool(true));
                slot.1.set("source", J::s("taught"));
                slot.1.set("blocked", J::Bool(false));
                slot.1.set("ts", J::n(now()));
            } else {
                cache.push((key.to_string(), entry));
                inserted = true;
            }
        }
        if inserted {
            if let Some(idx) = &mut self.idx { idx.add(key, qv); }
            else { self.ensure_index(); }
        }
    }

    /// Tipli öğretme: koleksiyon/tool/slot/ttl ile niyet tanımla (kural/arac vb.).
    /// `kesin` dışı tiplerde cevap genelde boştur (host üretir/çağırır).
    pub fn define(&mut self, query: &str, answer: &str, meta: &Meta) -> PutOutcome {
        self.put_intent(query, answer, "taught", true, true, 1.0, meta)
    }

    /// Model kaynaklı niyet ekle (korunmaz; budama hedefi) — toplu/senaryo için.
    pub fn note(&mut self, query: &str, answer: &str) {
        self.put_intent(query, answer, "model", true, false, 0.0, &Meta::default());
    }

    /// Niyeti tamamen sil (GUI delete_intent ile aynı; CLI `forget` için).
    pub fn forget(&mut self, query: &str) -> bool {
        self.delete_intent(query)
    }

    // ---- çalışma-anı güven kolu (config.policy) ----
    pub fn policy(&self) -> Policy {
        let p = self.raw.get("config").and_then(|c| c.get("policy"));
        Policy {
            mode: p.and_then(|p| p.get("mod")).and_then(|v| v.as_str()).unwrap_or("karma").to_string(),
            esik: p.and_then(|p| p.get("esik")).and_then(|v| v.as_f64()).unwrap_or(0.65),
            ogrenme: p.and_then(|p| p.get("ogrenme")).map_or(true, |v| v.as_bool()),
        }
    }
    pub fn set_policy(&mut self, mode: Option<&str>, esik: Option<f64>, ogrenme: Option<bool>) {
        let cur = self.policy();
        let mode = mode.unwrap_or(&cur.mode);
        let esik = esik.unwrap_or(cur.esik);
        let ogrenme = ogrenme.unwrap_or(cur.ogrenme);
        if self.raw.get("config").is_none() { self.raw.set("config", J::Obj(vec![])); }
        if let Some(cfg) = self.raw.get_mut("config") {
            cfg.set("policy", J::Obj(vec![
                ("mod".into(), J::s(mode)),
                ("esik".into(), J::n(esik)),
                ("ogrenme".into(), J::Bool(ogrenme)),
            ]));
        }
    }

    /// SADECE HAFIZADAN getir — model ASLA çağrılmaz (geriye-uyum; eski API).
    pub fn recall(&self, query: &str) -> Option<Hit> {
        match self.best_intent(query) {
            Some(h) if h.sim >= REUSE_SIM && h.verified && !h.blocked => Some(h),
            _ => None,
        }
    }

    /// YENİ pipeline: tipli direktif döndürür. Politikayı (mod/eşik) UYGULAR,
    /// koleksiyona göre yönlendirir, TTL tazeliğini kontrol eder, slot çıkarır.
    /// Model ASLA çağrılmaz — "neyi hatırlarım/bakarım/yaparım" kararını verir.
    pub fn resolve(&self, query: &str) -> Directive {
        self.resolve_scoped(query, None)
    }

    /// Scope-aware resolve. `user=None` → yalnız global (anonim/gizli varsayılan);
    /// `Some("user:x")` → global + o kullanıcının kişisel niyetleri.
    /// En yakın İKİ adayı tek taramada döndür (2. aday marj güvenliği için).
    pub fn best_two_scoped(&self, query: &str, allowed: Option<&str>) -> (Option<Hit>, f64, String) {
        let qv = self.embedder.embed(query);
        self.best_two_vec(&qv, Some(&embed::key(query)), allowed)
    }

    /// Vektör-tabanlı en-yakın iki aday (KİP-AGNOSTİK: metin/veri/görüntü/ses hepsi
    /// buraya bir vektör olarak gelir). `exact_key` verilirse aday kümesine eklenir.
    pub fn best_two_vec(&self, qv: &[f64], exact_key: Option<&str>, allowed: Option<&str>)
        -> (Option<Hit>, f64, String) {
        let cache = match self.raw.get("cache") { Some(J::Obj(m)) => m, _ => return (None, -1.0, String::new()) };
        let cand: Option<HashSet<String>> = self.idx.as_ref().map(|ix| {
            let mut c = ix.candidates(qv);
            if let Some(k) = exact_key { c.insert(k.to_string()); }
            c
        });
        let mut best: Option<Hit> = None;
        let mut second_sim = -1.0f64;
        let mut second_ans = String::new();
        for (k, c) in cache {
            if let Some(set) = &cand { if !set.contains(k) { continue; } }
            let sc = c.get("scope").and_then(|v| v.as_str()).unwrap_or("global");
            if let Some(u) = allowed { if sc != "global" && sc != u { continue; } }
            let vec = match c.get("vec") { Some(v) => read_vec(v), None => continue };
            if vec.is_empty() { continue; }
            let sim = embed::cosine(qv, &vec);
            let ans = c.get("answer").and_then(|a| a.as_str()).unwrap_or("").to_string();
            if best.as_ref().map_or(true, |b| sim > b.sim) {
                if let Some(b) = &best { second_sim = b.sim; second_ans = b.answer.clone(); }
                best = Some(Hit {
                    key: k.clone(), sim, answer: ans,
                    verified: c.get("verified").map_or(true, |v| v.as_bool()),
                    blocked: c.get("blocked").map_or(false, |v| v.as_bool()),
                    learned: c.get("learned").map_or(false, |v| v.as_bool()),
                    collection: c.get("collection").and_then(|v| v.as_str()).unwrap_or("kesin").to_string(),
                    tool: c.get("tool").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    slots: match c.get("slots") { Some(J::Arr(a)) =>
                        a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect(), _ => Vec::new() },
                    ttl_sec: c.get("ttl_sec").and_then(|v| v.as_f64()).unwrap_or(-1.0),
                    ts: c.get("ts").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    scope: sc.to_string(),
                });
            } else if sim > second_sim {
                second_sim = sim; second_ans = ans;
            }
        }
        (best, second_sim, second_ans)
    }

    pub fn resolve_scoped(&self, query: &str, user: Option<&str>) -> Directive {
        if self.policy().mode == "sadece_yz" {
            return Directive::Miss { sim: 0.0, reason: "mod=sadece_yz".into() };
        }
        let allowed = user.unwrap_or("global");   // anonim → yalnız global
        let (best, second_sim, second_ans) = self.best_two_scoped(query, Some(allowed));
        // Birebir (normalize) anahtar eşleşmesi = EN GÜÇLÜ sinyal (kullanıcı tam o
        // öğretilen soruyu sordu) → marj-çekimser atlanır (yoğun/benzer verisetinde
        // bile birebir hatırlama garantisi). Marj yalnız BULANIK eşleşme içindir.
        let exact = best.as_ref().map_or(false, |b| b.key == embed::key(query));
        self.directive_from(best, second_sim, second_ans, query, exact)
    }

    /// Vektör-tabanlı resolve (KİP-AGNOSTİK): host görüntü/ses/veriyi parmaklar,
    /// vektörü doğrudan sorar. Aynı güvenlik katmanı (eşik+marj+engel) uygulanır.
    pub fn resolve_vec(&self, qv: &[f64], user: Option<&str>) -> Directive {
        if self.policy().mode == "sadece_yz" {
            return Directive::Miss { sim: 0.0, reason: "mod=sadece_yz".into() };
        }
        let allowed = user.unwrap_or("global");
        let (best, second_sim, second_ans) = self.best_two_vec(qv, None, Some(allowed));
        // Medyada "birebir" = neredeyse birebir vektör (aynı parmak izi).
        let exact = best.as_ref().map_or(false, |b| b.sim >= 0.9999);
        self.directive_from(best, second_sim, second_ans, "", exact)
    }

    /// En-yakın adaylardan tipli direktif üret (metin/medya ortak son aşama).
    /// `slot_query` slot çıkarımı için ham metin (medyada boş). `exact`=birebir
    /// anahtar eşleşmesi (marj-çekimser atlanır).
    fn directive_from(&self, best: Option<Hit>, second_sim: f64, second_ans: String,
                      slot_query: &str, exact: bool) -> Directive {
        let pol = self.policy();
        let best = match best {
            Some(h) => h,
            None => return Directive::Miss { sim: 0.0, reason: "boş beyin".into() },
        };
        let sim = best.sim;
        if best.blocked {
            return Directive::Miss { sim, reason: "engelli niyet".into() };
        }
        // Eşik: sadece_mimar'da düşük bar (bileni ver), karma'da güven eşiği.
        let bar = if pol.mode == "sadece_mimar" { REUSE_SIM } else { REUSE_SIM.max(pol.esik) };
        if sim < bar || !best.verified {
            return Directive::Miss { sim, reason: format!("eşik altı ({:.2}<{:.2})", sim, bar) };
        }
        // MARJ ÇEKİMSER: en yakın iki aday birbirine yakın VE FARKLI cevaplıysa,
        // hangisinin doğru olduğu belirsizdir → servis etme, modele düş. Birebir/
        // duplike (aynı cevap) durumunda tetiklenmez. ("ağırlıklar çok yakın" riski)
        // BİREBİR anahtar eşleşmesinde atlanır (kesin sinyal > komşu benzerliği).
        if !exact && second_sim >= 0.0 && (sim - second_sim) < AMBIGUITY_MARGIN
            && second_ans != best.answer {
            return Directive::Miss { sim, reason:
                format!("belirsiz (marj {:.2}<{:.2}: iki yakın aday farklı cevap)",
                        sim - second_sim, AMBIGUITY_MARGIN) };
        }
        // Koleksiyona göre yönlendir.
        match best.collection.as_str() {
            "kural" => Directive::Rule {
                handler: best.tool.clone(),
                query: best.key.clone(),
                slots: extract_slots(slot_query, &best.slots), sim,
            },
            "arac" => Directive::Tool {
                tool: best.tool.clone(),
                slots: extract_slots(slot_query, &best.slots), sim, action: false,
            },
            "eylem" => Directive::Tool {
                tool: best.tool.clone(),
                slots: extract_slots(slot_query, &best.slots), sim, action: true,
            },
            "uretim" => Directive::Miss { sim, reason: "uretim: her sefer üret".into() },
            _ => {
                // kesin (ya da bilinmeyen) = gerçek. TTL tazeliği:
                if best.ttl_sec == 0.0 {
                    return Directive::Stale { collection: best.collection.clone(), sim };
                }
                if best.ttl_sec > 0.0 && (now() - best.ts) > best.ttl_sec {
                    return Directive::Stale { collection: best.collection.clone(), sim };
                }
                Directive::Fact { answer: best.answer.clone(), sim, learned: best.learned }
            }
        }
    }

    /// Çekirdek döngü: biliyorsa beyinden (ÜCRETSİZ), bilmiyorsa motoru çağır + öğren.
    /// (Ağ gerektirir → `net` özelliği; gömülü/wasm çekirdekte resolve/recall kullanılır.)
    #[cfg(feature = "net")]
    pub fn ask(&mut self, query: &str, engine: &Engine, risk_high: bool)
               -> Result<(String, String, f64), String> {
        let best = self.best_intent(query);
        if let Some(h) = &best {
            // risk=high: kestirmeyi eleme, her zaman güçlü model + doğrula.
            if !risk_high && h.sim >= REUSE_SIM && h.verified && !h.blocked {
                // Yeniden kullanım = bu niyetin pekiştirilmesi → ML'i de eğit.
                self.ml.train(&self.embedder.embed(query), &h.key);
                let src = if h.learned { "rule" } else { "cache" };
                return Ok((h.answer.clone(), src.to_string(), h.sim));
            }
        }
        let sim = best.as_ref().map_or(0.0, |b| b.sim);
        // Model gerekli → gerçek LLM
        let ans = engine.complete(&self.system_prompt(), query)?;
        if let Some(bad) = self.check_guard(&ans) {
            return Ok((format!("(Engellendi: '{}' yasak deseni)", bad),
                       "blocked".to_string(), sim));
        }
        self.put_intent(query, &ans, "model", true, false, 0.0, &Meta::default());
        Ok((ans, "model".to_string(), sim))
    }
}

#[derive(Clone)]
struct Ent {
    key: String,
    vec: Vec<f64>,
    n: f64,
    repeats: f64,
    hits: f64,
    reward: f64,
    ts: f64,
    answer: String,
    query: String,
    verified: bool,
    learned: bool,
    blocked: bool,
    source: String,
    collection: String,
    tool: String,
    slots: Vec<String>,
    ttl_sec: f64,
    volatility: String,
    scope: String,
}

impl Ent {
    fn to_json(&self) -> J {
        let mut e = J::Obj(vec![
            ("query".into(), J::s(&self.query)),
            ("answer".into(), J::s(&self.answer)),
            ("vec".into(), write_vec(&self.vec)),
            ("n".into(), J::n(self.n)),
            ("hits".into(), J::n(self.hits)),
            ("repeats".into(), J::n(self.repeats)),
            ("verified".into(), J::Bool(self.verified)),
            ("reward".into(), J::n(self.reward)),
            ("blocked".into(), J::Bool(self.blocked)),
            ("learned".into(), J::Bool(self.learned)),
            ("source".into(), J::s(&self.source)),
            ("ts".into(), J::n(self.ts)),
            ("collection".into(), J::s(&self.collection)),
            ("tool".into(), J::s(&self.tool)),
            ("slots".into(), J::Arr(self.slots.iter().map(|s| J::s(s)).collect())),
            ("ttl_sec".into(), J::n(self.ttl_sec)),
            ("volatility".into(), J::s(&self.volatility)),
            ("scope".into(), J::s(&self.scope)),
        ]);
        strip_defaults(&mut e);
        e
    }
}

// yardımcı: rules dizisine ekleme için
trait ArrMut { fn obj_entries_mut_arr(&mut self) -> Option<&mut Vec<J>>; }
impl ArrMut for J {
    fn obj_entries_mut_arr(&mut self) -> Option<&mut Vec<J>> {
        if let J::Arr(a) = self { Some(a) } else { None }
    }
}

// ----------------------------- kapsayıcı ----------------------------------
fn read_container(path: &str) -> Result<J, String> {
    let raw = std::fs::read(path).map_err(|e| format!("okunamadı: {}", e))?;
    parse_container(&raw)
}

/// Bayt tamponundan GÜVENLİ ayrıştırma — çöp/kesik/kötü-niyetli girdide ASLA panik
/// etmez (sınır kontrollü), Err döner. Sürüm göçü: eski sürüm kabul (JSON gövdesi
/// kendini tanımlar), daha yeni sürüm → açık hata (motoru güncelle).
pub fn parse_container(raw: &[u8]) -> Result<J, String> {
    if raw.len() < 12 || &raw[0..4] != MAGIC {
        return Err("geçerli bir Mimar beyni değil (MAGIC yok)".into());
    }
    let ver = u16::from_le_bytes([raw[4], raw[5]]);
    if ver > FORMAT_VERSION {
        return Err(format!("dosya sürümü {} bu motordan ({}) yeni — Mimar'ı güncelleyin",
                           ver, FORMAT_VERSION));
    }
    let json_len = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize;
    let end = 12usize.checked_add(json_len).ok_or("bozuk başlık (json_len taşması)")?;
    if end > raw.len() {
        return Err(format!("bozuk/kesik dosya (json_len {} > kalan {} bayt)",
                           json_len, raw.len().saturating_sub(12)));
    }
    json::parse(&raw[12..end])
}

fn entry_score(e: &J) -> (f64, f64) {
    (e.get("reward").and_then(|v| v.as_f64()).unwrap_or(0.0),
     e.get("repeats").and_then(|v| v.as_f64()).unwrap_or(0.0))
}

/// İki+ beyni provenanslı tek beyne birleştir. Çakışma (aynı anahtar) ödül/tekrar
/// ile çözülür; guard'lar toplamsaldır; her kaynak içerik-hash'iyle kaydedilir.
pub fn merge_brains(sources: &[Brain]) -> J {
    let mut raw = new_data("hash-256");
    let roles: Vec<String> = sources.iter().map(|b| b.role())
        .filter(|r| !r.is_empty() && r != "Yardımcı bir asistan.").collect();
    if !roles.is_empty() { raw.set("role", J::s(&roles.join("\n---\n"))); }

    let mut prov = Vec::new();
    let mut cache_out: Vec<(String, J)> = Vec::new();
    let mut rules_out: Vec<J> = Vec::new();
    let mut rule_seen: Vec<String> = Vec::new();

    for (i, b) in sources.iter().enumerate() {
        prov.push(J::Obj(vec![
            ("tag".into(), J::s(&format!("S{}", i + 1))),
            ("source".into(), J::s(b.path.as_deref().unwrap_or("bellek"))),
            ("hash".into(), J::s(&b.content_hash())),
            ("intents".into(), J::n(b.cache_len() as f64)),
            ("ts".into(), J::n(now())),
        ]));
        if let Some(J::Obj(c)) = b.raw.get("cache") {
            for (k, ent) in c {
                if let Some(pos) = cache_out.iter().position(|(kk, _)| kk == k) {
                    if entry_score(ent) > entry_score(&cache_out[pos].1) {
                        cache_out[pos].1 = ent.clone();
                    }
                } else {
                    cache_out.push((k.clone(), ent.clone()));
                }
            }
        }
        if let Some(J::Arr(rs)) = b.raw.get("rules") {
            for r in rs {
                let sig = format!("{:?}|{:?}|{:?}",
                    r.get("kind").and_then(|v| v.as_str()),
                    r.get("text").and_then(|v| v.as_str()),
                    r.get("forbid").and_then(|v| v.as_str()));
                if !rule_seen.contains(&sig) {
                    rule_seen.push(sig);
                    rules_out.push(r.clone());
                }
            }
        }
    }
    raw.set("cache", J::Obj(cache_out));
    raw.set("rules", J::Arr(rules_out));
    raw.set("provenance", J::Arr(prov));
    raw
}

pub fn save_raw(path: &str, raw: &J) -> Result<(), String> {
    write_container(path, raw)
}

fn write_container(path: &str, data: &J) -> Result<(), String> {
    let body = json::write(data).into_bytes();
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // flags
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out.extend_from_slice(&0u32.to_le_bytes()); // nblob (Faz 1: blob yok)
    let tmp = format!("{}.tmp", path);
    std::fs::write(&tmp, &out).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_extraction_custom_and_builtin() {
        // Özel slot adları: "ad değer", "ad=değer", "ad: değer"
        let s = extract_slots("rezervasyonu iptal et rezervasyon_no 4471", &["rezervasyon_no".into()]);
        assert_eq!(s, vec![("rezervasyon_no".to_string(), "4471".to_string())]);
        let s = extract_slots("iptal rezervasyon_no=8820", &["rezervasyon_no".into()]);
        assert_eq!(s, vec![("rezervasyon_no".to_string(), "8820".to_string())]);
        let s = extract_slots("sil kod: ABC12", &["kod".into()]);
        assert_eq!(s, vec![("kod".to_string(), "ABC12".to_string())]);
        // Çoklu özel slot
        let s = extract_slots("transfer kimden ahmet kime mehmet", &["kimden".into(), "kime".into()]);
        assert_eq!(s, vec![("kimden".to_string(), "ahmet".to_string()),
                           ("kime".to_string(), "mehmet".to_string())]);
        // Yerleşik sayı slotu hâlâ çalışır
        let s = extract_slots("oda 3 kişi", &["kisi".into()]);
        assert_eq!(s, vec![("kisi".to_string(), "3".to_string())]);
        // Slot adı yoksa boş (yanlış değer uydurmaz)
        assert!(extract_slots("rezervasyonu iptal et", &["rezervasyon_no".into()]).is_empty());
    }

    #[test]
    fn prop_save_load_roundtrip_identity() {
        // DEĞİŞMEZ: rastgele öğretilen beyin kaydedilip açılınca aynı niyet/cevaplar döner.
        let mut seed = 0xA5A5u64;
        for _ in 0..40 {
            let tmp = std::env::temp_dir().join(format!("mimar_prop_{}.mimar", seed));
            let _ = std::fs::remove_file(&tmp);
            let mut want: Vec<(String, String)> = Vec::new();
            { let mut b = Brain::open(Some(tmp.to_str().unwrap())).unwrap();
              let k = 1 + (seed % 30) as usize;
              for i in 0..k {
                  seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                  let q = format!("soru {}-{}", seed % 1000, i);
                  let a = format!("cevap {}", seed % 7);
                  b.teach(&q, &a); want.push((q, a));
              }
              b.save().unwrap(); }
            // yeniden aç → her öğretilen soru aynı cevapla birebir dönmeli
            let b2 = Brain::open(Some(tmp.to_str().unwrap())).unwrap();
            for (q, a) in &want {
                match b2.resolve(q) {
                    Directive::Fact { answer, .. } => assert_eq!(&answer, a,
                        "round-trip cevabı değişti: {}", q),
                    other => panic!("round-trip sonrası birebir soru Fact dönmeli, döndü: {:?} ({})", other, q),
                }
            }
            let _ = std::fs::remove_file(&tmp);
            seed = seed.wrapping_add(0x9E3779B9);
        }
    }

    #[test]
    fn prop_taught_exact_always_resolves() {
        // DEĞİŞMEZ: ne kadar BENZER/yoğun niyet olursa olsun, öğretilen sorunun BİREBİR
        // kendisi daima o cevabı döndürür (birebir-anahtar marj-çekimseri atlar).
        let mut b = Brain::open(None).unwrap();
        let n = 400;
        for i in 0..n { b.teach(&format!("kayit numarasi {}", i), &format!("deger {}", i)); }
        let mut seed = 12345u64;
        for _ in 0..100 {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            let i = (seed % n as u64) as usize;
            match b.resolve(&format!("kayit numarasi {}", i)) {
                Directive::Fact { answer, .. } => assert_eq!(answer, format!("deger {}", i)),
                other => panic!("birebir soru '{}' Fact dönmeli, döndü: {:?}", i, other),
            }
        }
    }

    #[test]
    fn concurrency_stress_no_corruption() {
        // RwLock<Brain> altında eşzamanlı okuma+yazma: deadlock/bozulma olmamalı,
        // tüm yazmalar tutmalı (httpd'nin paylaşım modeli).
        use std::sync::{Arc, RwLock};
        use std::thread;
        let brain = Arc::new(RwLock::new(Brain::open(None).unwrap()));
        let mut hs = Vec::new();
        // 8 yazıcı × 100 niyet
        for t in 0..8 {
            let b = Arc::clone(&brain);
            hs.push(thread::spawn(move || {
                for i in 0..100 {
                    b.write().unwrap().teach(&format!("soru {} {}", t, i), &format!("cevap {} {}", t, i));
                }
            }));
        }
        // 8 okuyucu eşzamanlı resolve (çökmesin)
        for _ in 0..8 {
            let b = Arc::clone(&brain);
            hs.push(thread::spawn(move || {
                for i in 0..200 { let _ = b.read().unwrap().resolve(&format!("soru 0 {}", i % 100)); }
            }));
        }
        for h in hs { h.join().unwrap(); }
        assert_eq!(brain.read().unwrap().cache_len(), 800, "8×100 yazma eksiksiz tutmalı");
    }

    #[test]
    fn large_scale_resolve_correct() {
        // Büyük-ölçek: 5000 niyet → HNSW aktif, birebir sorgular doğru dönmeli.
        let mut b = Brain::open(None).unwrap();
        for i in 0..5000 { b.teach(&format!("benzersiz kayit numarasi {}", i), &format!("deger {}", i)); }
        assert_eq!(b.cache_len(), 5000);
        // Rastgele 50 birebir sorgu → doğru cevap
        let mut ok = 0;
        for &i in &[0usize, 1, 42, 100, 999, 2500, 4999, 3333, 1234, 777] {
            if let Directive::Fact { answer, .. } = b.resolve(&format!("benzersiz kayit numarasi {}", i)) {
                if answer == format!("deger {}", i) { ok += 1; }
            }
        }
        assert_eq!(ok, 10, "büyük ölçekte birebir sorgular doğru dönmeli (HNSW)");
    }

    #[test]
    fn fuzz_parser_never_panics() {
        // Çöp/rastgele/kesik baytlar ayrıştırıcıda ASLA panik yaratmamalı (Err döner).
        let mut seed = 0xDEADBEEFu64;
        for _ in 0..20000 {
            let mut x = seed; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; seed = x;
            let len = (x % 200) as usize;
            let buf: Vec<u8> = (0..len).map(|i| {
                let mut y = seed.wrapping_add(i as u64); y ^= y >> 7; (y & 0xff) as u8
            }).collect();
            let _ = parse_container(&buf);     // panik etmemeli
            let _ = json::parse(&buf);         // panik etmemeli
        }
        // Bazıları MAGIC ile başlasın (daha derin yollar)
        for _ in 0..20000 {
            let mut x = seed; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; seed = x;
            let len = (x % 120) as usize;
            let mut buf = MAGIC.to_vec();
            buf.extend((0..len).map(|i| ((seed.wrapping_add(i as u64)) & 0xff) as u8));
            let _ = parse_container(&buf);
        }
    }

    #[test]
    fn truncation_and_bad_header_return_err_not_panic() {
        // Geçerli bir beyni üret, HER prefiks uzunluğunda kes → panik yok.
        let mut b = Brain::open(None).unwrap();
        b.teach("soru", "cevap");
        let body = json::write(&b.raw).into_bytes();
        let mut full = MAGIC.to_vec();
        full.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        full.extend_from_slice(&0u16.to_le_bytes());
        full.extend_from_slice(&(body.len() as u32).to_le_bytes());
        full.extend_from_slice(&body);
        for cut in 0..full.len() {
            let _ = parse_container(&full[..cut]);   // panik etmemeli
        }
        // json_len alanını DEVASA yap (eski panik bug'ı) → Err, panik değil
        let mut evil = full.clone();
        evil[8..12].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        assert!(parse_container(&evil).is_err(), "şişirilmiş json_len Err dönmeli");
        // Daha yeni sürüm → açık hata
        let mut newer = full.clone();
        newer[4..6].copy_from_slice(&(FORMAT_VERSION + 9).to_le_bytes());
        assert!(parse_container(&newer).is_err(), "yeni sürüm reddedilmeli");
        // Tam dosya hâlâ düzgün ayrışır
        assert!(parse_container(&full).is_ok(), "sağlam dosya ayrışmalı");
    }

    #[test]
    fn management() {
        let mut b = Brain::open(None).unwrap();
        b.teach("soru bir", "cevap bir");
        assert!(b.edit_answer("soru bir", "yeni cevap"));
        assert!(b.intent_rows().iter().any(|(q, a, _, _, _)| q == "soru bir" && a == "yeni cevap"));
        assert!(b.set_blocked("soru bir", true));
        assert!(b.intent_rows().iter().any(|(q, _, _, _, bl)| q == "soru bir" && *bl));
        assert!(b.delete_intent("soru bir"));
        assert!(b.intent_rows().is_empty());
        b.add_guard("yasak");
        assert_eq!(b.guard_patterns(), vec!["yasak".to_string()]);
        b.remove_guard("yasak");
        assert!(b.guard_patterns().is_empty());
        b.add_memory("bir bilgi", "fact");
        assert_eq!(b.memory_rows().len(), 1);
        b.clear_memory();
        assert!(b.memory_rows().is_empty());
    }

    #[test]
    fn teach_upsert_overrides_taught() {
        // P1: aynı soruyu yeniden öğretmek eski TAUGHT cevabı EZMELİ (yalancı başarı değil).
        let mut b = Brain::open(None).unwrap();
        assert_eq!(b.teach("kod adı nedir", "Terra"), PutOutcome::Created);
        assert_eq!(b.teach("kod adı nedir", "Konak"), PutOutcome::Updated);
        let h = b.best_intent("kod adı nedir").unwrap();
        assert_eq!(h.answer, "Konak");
        assert!(h.learned && h.verified);
        assert_eq!(b.cache_len(), 1); // çift kayıt oluşmamalı
    }

    #[test]
    fn teach_promotes_model_entry() {
        // P2: model kaydının üstüne teach gelince taught'a TERFİ etmeli.
        let mut b = Brain::open(None).unwrap();
        b.note("liman kaç", "8080"); // model kaynaklı
        assert_eq!(b.teach("liman kaç", "5432"), PutOutcome::Updated);
        let rows = b.intent_rows();
        let row = rows.iter().find(|(q, ..)| q == "liman kaç").unwrap();
        assert_eq!(row.1, "5432");
        assert_eq!(row.2, "taught"); // source terfi etti
    }

    #[test]
    fn model_note_never_overrides_taught() {
        // Otomatik öğrenme (model) öğretilmiş ground-truth'u EZMEMELİ.
        let mut b = Brain::open(None).unwrap();
        b.teach("veritabanı hangisi", "PostgreSQL");
        b.note("veritabanı hangisi", "MySQL");
        let h = b.best_intent("veritabanı hangisi").unwrap();
        assert_eq!(h.answer, "PostgreSQL");
    }

    #[test]
    fn forget_removes_intent() {
        let mut b = Brain::open(None).unwrap();
        b.teach("geçici", "sil beni");
        assert!(b.forget("geçici"));
        assert!(b.best_intent("geçici").is_none());
    }

    #[test]
    fn kesin_fact_resolves() {
        let mut b = Brain::open(None).unwrap();
        b.teach("check-in saati kaçta", "14:00");
        match b.resolve("check-in saati kaçta") {
            Directive::Fact { answer, .. } => assert_eq!(answer, "14:00"),
            d => panic!("fact bekleniyordu: {:?}", d),
        }
    }

    #[test]
    fn kural_extracts_slot_not_frozen_answer() {
        // 15≠16 bug'ı: kural slot çıkarır, donmuş cevap vermez.
        let mut b = Brain::open(None).unwrap();
        let meta = Meta { collection: Some("kural".into()), slots: vec!["saat".into()], ..Default::default() };
        b.define("girişi saat kaçta yapabilirim", "", &meta);
        let d15 = b.resolve("girişi saat 15 te yapabilirim");
        let d16 = b.resolve("girişi saat 16 da yapabilirim");
        match (d15, d16) {
            (Directive::Rule { slots: s15, .. }, Directive::Rule { slots: s16, .. }) => {
                assert_eq!(s15, vec![("saat".into(), "15".into())]);
                assert_eq!(s16, vec![("saat".into(), "16".into())]);  // FARKLI slot → farklı sonuç
            }
            other => panic!("iki Rule bekleniyordu: {:?}", other),
        }
    }

    #[test]
    fn arac_routes_to_tool_never_caches() {
        let mut b = Brain::open(None).unwrap();
        let meta = Meta { collection: Some("arac".into()), tool: Some("musaitlik".into()),
                          slots: vec!["tarih".into()], ttl_sec: Some(0.0), ..Default::default() };
        b.define("tarihinde boş oda var mı", "", &meta);
        match b.resolve("25 Ekim tarihinde boş oda var mı") {
            Directive::Tool { tool, slots, .. } => {
                assert_eq!(tool, "musaitlik");
                assert_eq!(slots, vec![("tarih".into(), "25 ekim".into())]);
            }
            d => panic!("tool bekleniyordu: {:?}", d),
        }
    }

    #[test]
    fn policy_sadece_yz_bypasses_memory() {
        let mut b = Brain::open(None).unwrap();
        b.teach("check-in", "14:00");
        b.set_policy(Some("sadece_yz"), None, None);
        assert!(matches!(b.resolve("check-in"), Directive::Miss { .. }));
        b.set_policy(Some("karma"), None, None);
        assert!(matches!(b.resolve("check-in"), Directive::Fact { .. }));
    }

    #[test]
    fn redact_masks_pii() {
        let r = redact("adım Mehmet telefonum 0555 123 45 67 eposta a.b@mail.com");
        assert!(r.contains("[telefon]"), "telefon maskelenmeli: {}", r);
        assert!(r.contains("[e-posta]"), "eposta maskelenmeli: {}", r);
        assert!(r.contains("Mehmet"));   // ad maskelenmez (PII tespiti değil, format tabanlı)
        assert!(!r.contains("0555"));
    }

    #[test]
    fn scope_isolation() {
        let mut b = Brain::open(None).unwrap();
        let m = Meta { scope: Some("user:ali".into()), ..Default::default() };
        b.define("en sevdiğim renk nedir", "mavi", &m);
        // ali görür
        assert!(matches!(b.resolve_scoped("en sevdiğim renk nedir", Some("user:ali")),
                         Directive::Fact { .. }));
        // veli GÖRMEZ (başka kullanıcı)
        assert!(matches!(b.resolve_scoped("en sevdiğim renk nedir", Some("user:veli")),
                         Directive::Miss { .. }));
        // anonim/global GÖRMEZ
        assert!(matches!(b.resolve("en sevdiğim renk nedir"), Directive::Miss { .. }));
    }

    #[test]
    fn city_slot_turkish_fold() {
        assert_eq!(slot_city("İstanbul hava durumu nasıl"), Some("istanbul".into()));
        assert_eq!(slot_city("çanakkale için hava"), Some("çanakkale".into()));
        assert_eq!(slot_city("hiç şehir yok burada"), None);
    }

    #[test]
    fn policy_high_threshold_abstains() {
        let mut b = Brain::open(None).unwrap();
        b.teach("otopark var mı ücretli mi acaba", "ücretsiz");
        // Gevşek eşleşen farklı soru — yüksek eşikte çekimser kalmalı.
        b.set_policy(Some("karma"), Some(0.99), None);
        assert!(matches!(b.resolve("araba park yeri"), Directive::Miss { .. }));
    }
}
