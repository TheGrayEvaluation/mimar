//! Uzak Mimar istemcisi — çalışan `mimar serve`'e REST/HTTP ile bağlan (MySQL
//! Workbench'in sunucuya bağlanması gibi). Saf std TCP (düz HTTP, httpd ile eşleşir),
//! sıfır bağımlılık. GUI uzak modu + script/otomasyon bunu kullanır.
//!
//! TLS YOK (yerel/İÇ ağ içindir); internete açılacaksa ters-proxy/TLS tüneli ardına
//! konur. Yazma işlemleri sunucu token isterse `token` ile Bearer gönderilir.

use crate::json::{self, J};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

pub struct Remote {
    host: String,
    port: u16,
    token: String,
}

impl Remote {
    /// base: "http://127.0.0.1:7700" veya "127.0.0.1:7700".
    pub fn new(base: &str, token: &str) -> Self {
        let b = base.trim().trim_end_matches('/');
        let b = b.strip_prefix("http://").unwrap_or(b);
        let (host, port) = match b.rsplit_once(':') {
            Some((h, p)) => (h.to_string(), p.parse().unwrap_or(7700)),
            None => (b.to_string(), 7700),
        };
        Remote { host, port, token: token.to_string() }
    }

    fn request(&self, method: &str, path: &str, body: Option<&str>) -> Result<String, String> {
        let payload = body.unwrap_or("");
        let mut head = format!(
            "{} {} HTTP/1.1\r\nHost: {}:{}\r\nConnection: close\r\n",
            method, path, self.host, self.port);
        if !self.token.is_empty() {
            head.push_str(&format!("Authorization: Bearer {}\r\n", self.token));
        }
        if method == "POST" {
            head.push_str("Content-Type: application/json\r\n");
            head.push_str(&format!("Content-Length: {}\r\n", payload.as_bytes().len()));
        }
        head.push_str("\r\n");
        let req = format!("{}{}", head, payload);

        let mut s = TcpStream::connect(format!("{}:{}", self.host, self.port))
            .map_err(|e| format!("bağlanılamadı {}:{} — {}", self.host, self.port, e))?;
        s.set_read_timeout(Some(Duration::from_secs(15))).ok();
        s.set_write_timeout(Some(Duration::from_secs(15))).ok();
        s.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
        let mut raw = Vec::new();
        s.read_to_end(&mut raw).map_err(|e| e.to_string())?;
        let text = String::from_utf8_lossy(&raw);
        let hdr_end = text.find("\r\n\r\n").ok_or("bozuk HTTP yanıtı")?;
        let status_line = text.lines().next().unwrap_or("");
        let code: u16 = status_line.split_whitespace().nth(1)
            .and_then(|c| c.parse().ok()).unwrap_or(0);
        let mut bodytext = text[hdr_end + 4..].to_string();
        if text[..hdr_end].to_lowercase().contains("transfer-encoding: chunked") {
            bodytext = dechunk(&bodytext);
        }
        if code >= 400 {
            return Err(format!("sunucu {}: {}", code, bodytext.trim()));
        }
        Ok(bodytext)
    }

    pub fn stats(&self) -> Result<J, String> {
        json::parse(self.request("GET", "/stats", None)?.as_bytes()).map_err(|e| e.to_string())
    }
    pub fn health(&self) -> Result<J, String> {
        json::parse(self.request("GET", "/saglik", None)?.as_bytes()).map_err(|e| e.to_string())
    }
    /// Niyet listesi: (query, answer, source, learned, blocked).
    pub fn list(&self) -> Result<Vec<(String, String, String, bool, bool)>, String> {
        let j = json::parse(self.request("GET", "/list", None)?.as_bytes()).map_err(|e| e.to_string())?;
        let mut out = Vec::new();
        if let J::Arr(a) = j {
            for it in a {
                out.push((
                    it.get("query").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    it.get("answer").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    it.get("source").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    it.get("learned").map_or(false, |v| v.as_bool()),
                    it.get("blocked").map_or(false, |v| v.as_bool()),
                ));
            }
        }
        Ok(out)
    }
    pub fn resolve(&self, query: &str) -> Result<J, String> {
        let b = json::write(&J::Obj(vec![("query".into(), J::s(query))]));
        json::parse(self.request("POST", "/resolve", Some(&b))?.as_bytes()).map_err(|e| e.to_string())
    }
    pub fn teach(&self, prompt: &str, answer: &str) -> Result<String, String> {
        let b = json::write(&J::Obj(vec![("prompt".into(), J::s(prompt)), ("answer".into(), J::s(answer))]));
        self.request("POST", "/teach", Some(&b))
    }
    pub fn forget(&self, query: &str) -> Result<String, String> {
        let b = json::write(&J::Obj(vec![("query".into(), J::s(query))]));
        self.request("POST", "/forget", Some(&b))
    }
    pub fn policy(&self, mode: Option<&str>, esik: Option<f64>, ogrenme: Option<bool>) -> Result<String, String> {
        let mut o = vec![];
        if let Some(m) = mode { o.push(("mod".to_string(), J::s(m))); }
        if let Some(e) = esik { o.push(("esik".to_string(), J::n(e))); }
        if let Some(g) = ogrenme { o.push(("ogrenme".to_string(), J::Bool(g))); }
        self.request("POST", "/policy", Some(&json::write(&J::Obj(o))))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn remote_client_end_to_end() {
        // In-process httpd sunucusu başlat, gerçek Remote istemcisiyle konuş.
        let tmp = std::env::temp_dir().join("mimar_remote_test.mimar");
        let _ = std::fs::remove_file(&tmp);
        let _ = std::fs::remove_file(format!("{}.wal", tmp.display()));
        {
            let mut b = crate::store::Brain::open(Some(tmp.to_str().unwrap())).unwrap();
            b.teach("kargo ne zaman", "2-3 gün");
            b.save().unwrap();
        }
        let path = tmp.to_str().unwrap().to_string();
        let addr = "127.0.0.1:17755";
        thread::spawn(move || {
            let _ = crate::httpd::serve_http(&path, addr, false, "TOK");
        });
        thread::sleep(Duration::from_millis(400));

        let r = Remote::new("http://127.0.0.1:17755", "TOK");
        // list
        let rows = r.list().expect("list");
        assert_eq!(rows.len(), 1, "başlangıçta 1 niyet");
        // teach (token'lı yazma)
        r.teach("iade nasıl", "14 gün içinde").expect("teach");
        thread::sleep(Duration::from_millis(100));
        let rows2 = r.list().expect("list2");
        assert_eq!(rows2.len(), 2, "teach sonrası 2 niyet");
        assert!(rows2.iter().any(|(q, a, _, _, _)| q == "iade nasıl" && a == "14 gün içinde"),
                "yeni niyet doğru içerikle görünmeli");
        // resolve (sunucu beyninden, model çağırmadan)
        let d = r.resolve("iade nasıl").expect("resolve");
        assert_eq!(d.get("tip").and_then(|v| v.as_str()), Some("fact"));
        assert_eq!(d.get("cevap").and_then(|v| v.as_str()), Some("14 gün içinde"));
        // forget
        r.forget("iade nasıl").expect("forget");
        thread::sleep(Duration::from_millis(100));
        assert_eq!(r.list().unwrap().len(), 1, "forget sonrası 1 niyet");

        let _ = std::fs::remove_file(&tmp);
        let _ = std::fs::remove_file(format!("{}.wal", tmp.display()));
    }
}
