//! Takılabilir embedder — çekirdek VARSAYILANI hashing (dilden bağımsız, sıfır-dep).
//! İSTEYEN opt-in olarak nöral embedder takar (ör. Ollama bge-m3): eş-anlam + çapraz-dil
//! yakalanır. Bu bir embedder SEÇİMİDİR; çekirdeğe çeviri/dil mantığı girmez.
//! Hangi embedder'ın kullanıldığı beyin dosyasının `feature` alanında saklanır;
//! açılışta otomatik yeniden kurulur.

use crate::embed;
#[cfg(feature = "net")]
use crate::json::{self, J};
#[cfg(feature = "net")]
use std::io::{Read, Write};
#[cfg(feature = "net")]
use std::net::TcpStream;
#[cfg(feature = "net")]
use std::time::Duration;

pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Vec<f64>;
    fn dim(&self) -> usize;
    fn id(&self) -> String;
}

/// Varsayılan: bağımlılıksız hashing (256 boyut).
pub struct HashEmbedder;
impl Embedder for HashEmbedder {
    fn embed(&self, t: &str) -> Vec<f64> { embed::embed(t) }
    fn dim(&self) -> usize { embed::DIM }
    fn id(&self) -> String { "hash-256".into() }
}

/// Opt-in: Ollama embeddings (ör. bge-m3 çok-dilli). Saf std TCP/HTTP (yerel).
#[cfg(feature = "net")]
pub struct OllamaEmbedder {
    model: String,
    host: String,
    port: u16,
    dim: usize,
}
#[cfg(feature = "net")]
impl OllamaEmbedder {
    pub fn new(model: &str) -> Self {
        let mut e = OllamaEmbedder { model: model.into(), host: "127.0.0.1".into(), port: 11434, dim: 0 };
        e.dim = e.embed("boyut").len();   // boyutu bir kez prob ederek öğren
        e
    }
}
#[cfg(feature = "net")]
impl Embedder for OllamaEmbedder {
    fn embed(&self, t: &str) -> Vec<f64> { ollama_embed(&self.host, self.port, &self.model, t) }
    fn dim(&self) -> usize { self.dim }
    fn id(&self) -> String { format!("ollama:{}", self.model) }
}

/// Opt-in: GENEL/GENİŞ HTTP embedding istemcisi — tek bir sağlayıcıya kilitli DEĞİL.
/// Yanıt biçimlerinin ÇOĞUNU tanır (OpenAI, Ollama, Cohere, HuggingFace/TEI, Jina,
/// ham dizi…), böylece OpenAI'a özel değildir. API anahtarı ORTAM DEĞİŞKENİNDEN gelir
/// (beyne yazılmaz → dosya taşınabilir + sırlar güvende). İstek `{"model","input"}`.
#[cfg(feature = "cloud")]
pub struct CloudEmbedder { base: String, model: String, key: String, id: String, dim: usize }
#[cfg(feature = "cloud")]
impl CloudEmbedder {
    pub fn new(base: &str, model: &str, env_key: &str, id: &str) -> Self {
        let key = std::env::var(env_key).unwrap_or_default();
        let mut e = CloudEmbedder { base: base.trim_end_matches('/').into(),
            model: model.into(), key, id: id.into(), dim: 0 };
        e.dim = e.fetch("dimension").len();   // boyutu bir kez prob et
        e
    }
    fn fetch(&self, text: &str) -> Vec<f64> {
        let payload = json::write(&J::Obj(vec![
            ("model".into(), J::s(&self.model)), ("input".into(), J::s(text))]));
        // base zaten tam uç olabilir (…/embeddings) ya da taban (…/v1); ikisini de dene.
        let url = if self.base.ends_with("/embeddings") { self.base.clone() }
                  else { format!("{}/embeddings", self.base) };
        let mut req = ureq::post(&url).set("Content-Type", "application/json");
        if !self.key.is_empty() { req = req.set("Authorization", &format!("Bearer {}", self.key)); }
        let resp = req.timeout(Duration::from_secs(60)).send_string(&payload);
        let mut v = resp.ok()
            .and_then(|r| r.into_string().ok())
            .and_then(|body| json::parse(body.as_bytes()).ok())
            .map(|data| extract_embedding(&data))
            .unwrap_or_default();
        embed::normalize(&mut v);
        v
    }
}

/// Bir JSON yanıttan embedding vektörünü GENİŞ biçimde çıkar — bilinen tüm şekiller:
///   {"data":[{"embedding":[...]}]}   OpenAI / OpenRouter / Together / Azure
///   {"embeddings":[[...]]}           Ollama (/api/embed) / Cohere / Jina / Voyage
///   {"embedding":[...]}              Ollama legacy / bazı yerel sunucular
///   [[...]]                          HuggingFace TEI / ham matris
///   [...]                            ham vektör
#[cfg(any(feature = "net", feature = "cloud"))]
pub fn extract_embedding(data: &J) -> Vec<f64> {
    let as_vec = |j: &J| -> Vec<f64> {
        if let J::Arr(a) = j { a.iter().filter_map(|x| x.as_f64()).collect() } else { Vec::new() }
    };
    if let Some(J::Arr(d)) = data.get("data") {
        if let Some(e) = d.first().and_then(|x| x.get("embedding")) {
            let v = as_vec(e); if !v.is_empty() { return v; }
        }
    }
    if let Some(J::Arr(outer)) = data.get("embeddings") {
        if let Some(first) = outer.first() { let v = as_vec(first); if !v.is_empty() { return v; } }
    }
    if let Some(e) = data.get("embedding") { let v = as_vec(e); if !v.is_empty() { return v; } }
    if let J::Arr(outer) = data {
        if let Some(first) = outer.first() {           // [[...]]
            let v = as_vec(first); if !v.is_empty() { return v; }
        }
        let v = as_vec(data); if !v.is_empty() { return v; }  // [...]
    }
    Vec::new()
}
#[cfg(feature = "cloud")]
impl Embedder for CloudEmbedder {
    fn embed(&self, t: &str) -> Vec<f64> { self.fetch(t) }
    fn dim(&self) -> usize { self.dim }
    fn id(&self) -> String { self.id.clone() }
}

/// feature kimliğinden embedder kur. ÇEKİRDEK kendi ML'iyle (hashing) HER YERDE çalışır;
/// aşağıdaki nöral yollar tamamen OPSİYONEL ve SAĞLAYICI-BAĞIMSIZDIR (tek firmaya kilit yok):
///   `ollama:<model>`        → yerel Ollama (çevrimdışı)
///   `openai:<model>`        → OpenAI (`OPENAI_API_KEY`)
///   `openrouter:<model>`    → OpenRouter (`OPENROUTER_API_KEY`)
///   `http:<url>|<model>`    → HERHANGİ embeddings ucu (`EMBED_API_KEY`) — Cohere/HF/TEI/
///                             Jina/Voyage/kendi sunucun; yanıt biçimlerinin çoğu tanınır
///   (aksi) `data`           → yapısal veri · varsayılan → hashing (sıfır-dep, her yerde).
pub fn make_embedder(id: &str) -> Box<dyn Embedder> {
    #[cfg(feature = "net")]
    if let Some(m) = id.strip_prefix("ollama:") {
        return Box::new(OllamaEmbedder::new(m));
    }
    #[cfg(feature = "cloud")]
    {
        if let Some(m) = id.strip_prefix("openai:") {
            return Box::new(CloudEmbedder::new("https://api.openai.com/v1", m, "OPENAI_API_KEY", id));
        }
        if let Some(m) = id.strip_prefix("openrouter:") {
            return Box::new(CloudEmbedder::new("https://openrouter.ai/api/v1", m, "OPENROUTER_API_KEY", id));
        }
        // Genel: "http:<base-veya-tam-uç>|<model>"  (eski ad "cloud:" da kabul)
        for pfx in ["http:", "cloud:"] {
            if let Some(rest) = id.strip_prefix(pfx) {
                if let Some((base, model)) = rest.split_once('|') {
                    return Box::new(CloudEmbedder::new(base, model, "EMBED_API_KEY", id));
                }
            }
        }
    }
    if id == "data" || id == "data-256" {
        Box::new(crate::modality::DataEmbedder)   // yapısal veri kipi (native)
    } else {
        Box::new(HashEmbedder)
    }
}

#[cfg(feature = "net")]
fn ollama_embed(host: &str, port: u16, model: &str, text: &str) -> Vec<f64> {
    let payload = json::write(&J::Obj(vec![
        ("model".into(), J::s(model)), ("input".into(), J::s(text))]));
    let req = format!(
        "POST /api/embed HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        host, port, payload.as_bytes().len(), payload);
    let mut v = (|| -> Option<Vec<f64>> {
        let mut s = TcpStream::connect(format!("{}:{}", host, port)).ok()?;
        s.set_read_timeout(Some(Duration::from_secs(60))).ok();
        s.write_all(req.as_bytes()).ok()?;
        let mut raw = Vec::new();
        s.read_to_end(&mut raw).ok()?;
        let text = String::from_utf8_lossy(&raw);
        let bs = text.find("\r\n\r\n").map(|i| i + 4)?;
        let mut body = &text[bs..];
        let de;
        if text[..bs].to_lowercase().contains("transfer-encoding: chunked") {
            de = dechunk(body); body = &de;
        }
        let data = json::parse(body.as_bytes()).ok()?;
        Some(extract_embedding(&data))   // ortak GENİŞ çıkarıcı (çok biçim)
    })().unwrap_or_default();
    embed::normalize(&mut v);
    v
}

#[cfg(feature = "net")]
fn dechunk(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    loop {
        let nl = match rest.find("\r\n") { Some(i) => i, None => break };
        let size = usize::from_str_radix(rest[..nl].trim(), 16).unwrap_or(0);
        if size == 0 { break; }
        let start = nl + 2;
        if start + size > rest.len() { break; }
        out.push_str(&rest[start..start + size]);
        rest = &rest[start + size + 2..];
    }
    out
}

#[cfg(all(test, any(feature = "net", feature = "cloud")))]
mod tests {
    use super::extract_embedding;
    use crate::json;

    fn vec_of(body: &str) -> Vec<f64> {
        extract_embedding(&json::parse(body.as_bytes()).unwrap())
    }

    #[test]
    fn broad_extractor_handles_all_shapes() {
        // OpenAI / OpenRouter
        assert_eq!(vec_of(r#"{"data":[{"embedding":[1.0,2.0,3.0]}]}"#), vec![1.0,2.0,3.0]);
        // Ollama /api/embed, Cohere, Jina
        assert_eq!(vec_of(r#"{"embeddings":[[4.0,5.0]]}"#), vec![4.0,5.0]);
        // Ollama legacy
        assert_eq!(vec_of(r#"{"embedding":[6.0,7.0]}"#), vec![6.0,7.0]);
        // HuggingFace TEI / ham matris
        assert_eq!(vec_of(r#"[[8.0,9.0]]"#), vec![8.0,9.0]);
        // ham vektör
        assert_eq!(vec_of(r#"[10.0,11.0]"#), vec![10.0,11.0]);
        // tanınmayan → boş (çökmeden)
        assert!(vec_of(r#"{"baska":123}"#).is_empty());
    }
}
