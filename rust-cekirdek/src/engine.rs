//! Model motoru — yerel (Ollama / LM Studio, ham TCP/HTTP, SIFIR bağımlılık) VE
//! bulut (OpenRouter / OpenAI-uyumlu, HTTPS/TLS). TLS yalnız `cloud` özelliğinde
//! (ureq+rustls); çekirdek varsayılan derlemede sıfır bağımlılık kalır.

use crate::json::{self, J};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

pub struct Engine {
    pub host: String,
    pub port: u16,
    pub model: String,
    pub path: String,       // /api/chat (Ollama) | /chat/completions (OpenAI-uyumlu)
    pub openai: bool,
    pub https: bool,        // true → bulut (TLS); false → yerel ham TCP
    pub base_url: String,   // https modunda tam taban, ör. "https://openrouter.ai/api/v1"
    pub api_key: String,    // Bearer anahtar (bulut)
}

impl Engine {
    pub fn ollama(model: &str) -> Self {
        Engine { host: "127.0.0.1".into(), port: 11434, model: model.into(),
                 path: "/api/chat".into(), openai: false, https: false,
                 base_url: String::new(), api_key: String::new() }
    }
    pub fn openai_local(model: &str, port: u16) -> Self {
        Engine { host: "127.0.0.1".into(), port, model: model.into(),
                 path: "/v1/chat/completions".into(), openai: true, https: false,
                 base_url: String::new(), api_key: String::new() }
    }
    /// OpenRouter (bulut, HTTPS). `cloud` özelliğiyle derlenmeli.
    pub fn openrouter(model: &str, key: &str) -> Self {
        Engine { host: String::new(), port: 443, model: model.into(),
                 path: "/chat/completions".into(), openai: true, https: true,
                 base_url: "https://openrouter.ai/api/v1".into(), api_key: key.into() }
    }
    /// Herhangi bir OpenAI-uyumlu bulut (OpenAI/Groq/Together…). base_url: "https://…/v1".
    pub fn cloud(base_url: &str, model: &str, key: &str) -> Self {
        Engine { host: String::new(), port: 443, model: model.into(),
                 path: "/chat/completions".into(), openai: true, https: true,
                 base_url: base_url.trim_end_matches('/').into(), api_key: key.into() }
    }

    pub fn complete(&self, system: &str, user: &str) -> Result<String, String> {
        let msgs = J::Arr(vec![
            J::Obj(vec![("role".into(), J::s("system")), ("content".into(), J::s(system))]),
            J::Obj(vec![("role".into(), J::s("user")), ("content".into(), J::s(user))]),
        ]);
        let mut body = vec![
            ("model".to_string(), J::s(&self.model)),
            ("messages".to_string(), msgs),
            ("stream".to_string(), J::Bool(false)),
        ];
        if !self.openai {
            // Ollama: deterministikligi artir
            body.push(("options".to_string(),
                       J::Obj(vec![("temperature".into(), J::n(0.2))])));
        } else {
            body.push(("temperature".to_string(), J::n(0.2)));
        }
        let payload = json::write(&J::Obj(body));

        // Bulut (HTTPS/TLS) yolu — yalnız `cloud` özelliğinde.
        if self.https {
            return self.complete_https(&payload);
        }

        let req = format!(
            "POST {} HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{}",
            self.path, self.host, self.port, payload.as_bytes().len(), payload);

        let addr = format!("{}:{}", self.host, self.port);
        let mut stream = TcpStream::connect(&addr)
            .map_err(|e| format!("bağlanılamadı {}: {}", addr, e))?;
        stream.set_read_timeout(Some(Duration::from_secs(120))).ok();
        stream.write_all(req.as_bytes()).map_err(|e| e.to_string())?;

        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).map_err(|e| e.to_string())?;
        let text = String::from_utf8_lossy(&raw);
        let body_start = text.find("\r\n\r\n").map(|i| i + 4)
            .ok_or("HTTP gövdesi yok")?;
        let mut body_str = &text[body_start..];

        // Chunked transfer-encoding ise kabaca birleştir (Ollama Content-Length
        // verir ama garanti değil).
        let dechunked;
        if text[..body_start].to_lowercase().contains("transfer-encoding: chunked") {
            dechunked = dechunk(body_str);
            body_str = &dechunked;
        }

        let data = json::parse(body_str.as_bytes())
            .map_err(|e| format!("JSON çözülemedi: {} | gövde: {:.200}", e, body_str))?;

        // Ollama: {message:{content}}  |  OpenAI: {choices:[{message:{content}}]}
        if let Some(m) = data.get("message").and_then(|m| m.get("content")) {
            return Ok(m.as_str().unwrap_or("").to_string());
        }
        if let Some(J::Arr(ch)) = data.get("choices") {
            if let Some(c) = ch.first()
                .and_then(|c| c.get("message")).and_then(|m| m.get("content")) {
                return Ok(c.as_str().unwrap_or("").to_string());
            }
        }
        if let Some(e) = data.get("error") {
            return Err(format!("motor hatası: {:?}", e));
        }
        Err(format!("beklenmedik yanıt: {:.200}", body_str))
    }

    #[cfg(feature = "cloud")]
    fn complete_https(&self, payload: &str) -> Result<String, String> {
        let url = format!("{}{}", self.base_url, self.path);
        let resp = ureq::post(&url)
            .set("Content-Type", "application/json")
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("HTTP-Referer", "https://mimar.local")
            .set("X-Title", "Mimar")
            .timeout(Duration::from_secs(120))
            .send_string(payload);
        let body = match resp {
            Ok(r) => r.into_string().map_err(|e| e.to_string())?,
            Err(ureq::Error::Status(code, r)) => {
                let b = r.into_string().unwrap_or_default();
                return Err(format!("HTTP {} — {:.300}", code, b));
            }
            Err(e) => return Err(format!("bağlantı hatası: {}", e)),
        };
        let data = json::parse(body.as_bytes())
            .map_err(|e| format!("JSON çözülemedi: {} | {:.200}", e, body))?;
        if let Some(J::Arr(ch)) = data.get("choices") {
            if let Some(c) = ch.first()
                .and_then(|c| c.get("message")).and_then(|m| m.get("content")) {
                return Ok(c.as_str().unwrap_or("").to_string());
            }
        }
        if let Some(e) = data.get("error") {
            return Err(format!("API hatası: {:?}", e));
        }
        Err(format!("beklenmedik yanıt: {:.200}", body))
    }

    #[cfg(not(feature = "cloud"))]
    fn complete_https(&self, _payload: &str) -> Result<String, String> {
        Err("bulut/TLS için `cargo build --features cloud` ile derleyin".into())
    }
}

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
