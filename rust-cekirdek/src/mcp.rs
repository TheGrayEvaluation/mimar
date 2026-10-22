//! MCP sunucusu (Rust) — ajanların (Cursor / Antigravity / Windsurf) Mimar'ı
//! mount etmesi için. Saf std, satır-ayrımlı JSON-RPC 2.0 / stdio.
//!
//! Araçlar: mimar_ask (biliyorsa ücretsiz, bilmiyorsa yerel LLM), mimar_teach,
//! mimar_explain, mimar_feedback. Kaynak: mimar://stats.

use crate::engine::Engine;
use crate::json::{self, J};
use crate::store::{Brain, Meta};
use std::io::{BufRead, Write};

const PROTO: &str = "2024-11-05";

/// Bağlanan ajana (Cursor / Antigravity / Windsurf) Mimar'ın nasıl kullanılacağı.
/// initialize.instructions ile taşınır → yönlendirme sunucunun içindedir.
const INSTRUCTIONS: &str = "\
Bu projede bir Mimar beyni bağlı. Mimar; kullanıldıkça öğrenen, tekrar eden soruları \
modele sormadan ücretsiz/anında yanıtlayan kalıcı bir hafıza + karar katmanıdır. \
KURAL (kullanıcı söylemese de, her istekte): \
(1) Cevabı üretmeden ÖNCE 'mimar_resolve' çağır (JSON tipli direktif; model çağırmaz). \
(2) tip='fact' → cevabı doğrudan ver. tip='rule' → slotlarla kendi mantığını uygula. \
tip='tool' → ilgili fonksiyonu çağır. tip='stale' → taze veri çek. \
(3) tip='miss' → SEN cevapla, sonra tekrar edebilecek/önemli bilgiyi 'mimar_teach' ile kaydet. \
Değişken/parametrik bilgiyi değer olarak teach ETME; 'mimar_define' ile tiple (collection=arac/kural, tool, ttl=0). \
(4) Yanlış cevaba 'mimar_feedback' good=bad; kalıcı silmek 'mimar_forget'. \
(Basit kullanım için 'mimar_recall', mimar'ın kendi yerel modeli için 'mimar_ask' da var.) \
Böylece bilgi Mimar'da birikir, tutarlılık korunur ve tekrar eden iş MODEL ÇAĞRILMADAN ücretsiz gelir.";

fn tools() -> J {
    fn tool(name: &str, desc: &str, props: Vec<(&str, &str)>, req: Vec<&str>) -> J {
        let properties: Vec<(String, J)> = props.into_iter()
            .map(|(p, _t)| (p.to_string(), J::Obj(vec![("type".into(), J::s("string"))])))
            .collect();
        J::Obj(vec![
            ("name".into(), J::s(name)),
            ("description".into(), J::s(desc)),
            ("inputSchema".into(), J::Obj(vec![
                ("type".into(), J::s("object")),
                ("properties".into(), J::Obj(properties)),
                ("required".into(), J::Arr(req.into_iter().map(J::s).collect())),
            ])),
        ])
    }
    J::Arr(vec![
        tool("mimar_resolve", "TİPLİ yönlendirme (önerilen). JSON döner: fact=doğrudan kullan · \
rule=slotlarla host mantığı · tool=fonksiyon çağır · stale=yeniden çek · miss=sen cevapla+mimar_teach. \
Politika(mod/eşik)+koleksiyon+TTL+slot uygular; model çağırmaz.",
             vec![("query", "string")], vec!["query"]),
        tool("mimar_define", "Tipli niyet tanımla: collection=kesin|kural|arac|uretim|eylem, \
slots (virgüllü), tool (handler adı), ttl (sn; 0=asla cache), scope (user:<id>).",
             vec![("query", "string"), ("collection", "string"), ("slots", "string"),
                  ("tool", "string"), ("ttl", "string")], vec!["query", "collection"]),
        tool("mimar_recall", "SADECE hafızadan getir, model ÇAĞIRMAZ (basit; resolve'u tercih et). \
isabet (kaynak=rule/cache) varsa o cevabı ver; '[miss]' ise sen cevapla, sonra mimar_teach ile kaydet.",
             vec![("query", "string")], vec!["query"]),
        tool("mimar_ask", "Beyne sor: bilineni ücretsiz/anında, bilinmeyeni mimar'ın KENDİ yerel modeline yönlendirir.",
             vec![("query", "string")], vec!["query"]),
        tool("mimar_teach", "Beyne girdi→çıktı öğret (tekrar eden işi kalıcı derle).",
             vec![("prompt", "string"), ("answer", "string")], vec!["prompt", "answer"]),
        tool("mimar_explain", "Sorgu için planı göster (derlenmiş mi, model mi).",
             vec![("query", "string")], vec!["query"]),
        tool("mimar_feedback", "Cevaba iyi/kötü geri bildirim (kötü→niyet engellenir).",
             vec![("query", "string"), ("good", "string")], vec!["query", "good"]),
        tool("mimar_forget", "Bir niyeti kalıcı olarak sil (yanlış/eski bilgiyi temizle).",
             vec![("query", "string")], vec!["query"]),
    ])
}

fn text_result(text: &str) -> J {
    J::Obj(vec![
        ("content".into(), J::Arr(vec![J::Obj(vec![
            ("type".into(), J::s("text")), ("text".into(), J::s(text))])])),
        ("isError".into(), J::Bool(false)),
    ])
}

fn arg<'a>(params: &'a J, name: &str) -> &'a str {
    params.get("arguments").and_then(|a| a.get(name)).and_then(|v| v.as_str()).unwrap_or("")
}

fn handle(b: &mut Brain, engine: &Engine, msg: &J) -> Option<J> {
    let method = msg.get("method").and_then(|m| m.as_str())?;
    let id = msg.get("id").cloned();
    let result: J = match method {
        "initialize" => J::Obj(vec![
            ("protocolVersion".into(), J::s(PROTO)),
            ("capabilities".into(), J::Obj(vec![
                ("tools".into(), J::Obj(vec![])), ("resources".into(), J::Obj(vec![]))])),
            ("serverInfo".into(), J::Obj(vec![
                ("name".into(), J::s("mimar")), ("version".into(), J::s("0.1.0"))])),
            // Davranış protokolü SUNUCUDA taşınır: ajan kural dosyası olmasa bile (ör. global
            // kurulum) bağlanan ajan Mimar'ı nasıl kullanacağını buradan öğrenir.
            ("instructions".into(), J::s(INSTRUCTIONS)),
        ]),
        "notifications/initialized" | "initialized" => return None,
        "ping" => J::Obj(vec![]),
        "tools/list" => J::Obj(vec![("tools".into(), tools())]),
        "resources/list" => J::Obj(vec![("resources".into(), J::Arr(vec![
            J::Obj(vec![("uri".into(), J::s("mimar://stats")),
                        ("name".into(), J::s("Beyin istatistiği")),
                        ("mimeType".into(), J::s("application/json"))])]))]),
        "resources/read" => {
            let stats = format!("{{\"role\":\"{}\",\"intents\":{},\"guards\":{}}}",
                b.role().replace('"', "'"), b.cache_len(), b.guard_patterns().len());
            J::Obj(vec![("contents".into(), J::Arr(vec![J::Obj(vec![
                ("uri".into(), J::s("mimar://stats")),
                ("mimeType".into(), J::s("application/json")),
                ("text".into(), J::s(&stats))])]))])
        }
        "tools/call" => {
            let params = msg.get("params").cloned().unwrap_or(J::Null);
            let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
            match name {
                "mimar_resolve" => {
                    let d = b.resolve(arg(&params, "query"));
                    text_result(&json::write(&d.to_json()))
                }
                "mimar_define" => {
                    let slots: Vec<String> = arg(&params, "slots").split(',')
                        .map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
                    let tool = arg(&params, "tool");
                    let ttl = arg(&params, "ttl").parse::<f64>().ok();
                    let scope = arg(&params, "scope");
                    let meta = Meta {
                        collection: Some(arg(&params, "collection").to_string()),
                        tool: if tool.is_empty() { None } else { Some(tool.to_string()) },
                        slots, ttl_sec: ttl, volatility: None,
                        scope: if scope.is_empty() { None } else { Some(scope.to_string()) },
                    };
                    let o = b.define(arg(&params, "query"), arg(&params, "answer"), &meta);
                    let _ = b.save();
                    text_result(&format!("tanımlandı — {}", o.label()))
                }
                "mimar_recall" => {
                    match b.recall(arg(&params, "query")) {
                        Some(h) => {
                            let src = if h.learned { "rule" } else { "cache" };
                            text_result(&format!("{}\n\n[mimar] kaynak={} sim={:.3}", h.answer, src, h.sim))
                        }
                        None => text_result("[miss]\n\n[mimar] kaynak=miss (sen cevapla, sonra mimar_teach)"),
                    }
                }
                "mimar_ask" => {
                    match b.ask(arg(&params, "query"), engine, false) {
                        Ok((ans, src, sim)) => {
                            let _ = b.save();
                            text_result(&format!("{}\n\n[mimar] kaynak={} sim={:.3}", ans, src, sim))
                        }
                        Err(e) => text_result(&format!("(hata: {})", e)),
                    }
                }
                "mimar_teach" => {
                    let outcome = b.teach(arg(&params, "prompt"), arg(&params, "answer"));
                    let _ = b.save();
                    text_result(&format!("öğretildi — {}", outcome.label()))
                }
                "mimar_explain" => text_result(&b.explain(arg(&params, "query"), false)),
                "mimar_feedback" => {
                    let good = arg(&params, "good");
                    let msg = b.feedback(arg(&params, "query"), good == "good" || good == "iyi");
                    let _ = b.save();
                    text_result(&msg)
                }
                "mimar_forget" => {
                    let removed = b.forget(arg(&params, "query"));
                    let _ = b.save();
                    text_result(if removed { "unutuldu (niyet silindi)" } else { "eşleşen niyet yok" })
                }
                other => text_result(&format!("bilinmeyen araç: {}", other)),
            }
        }
        _ => {
            return id.map(|id| J::Obj(vec![
                ("jsonrpc".into(), J::s("2.0")), ("id".into(), id),
                ("error".into(), J::Obj(vec![
                    ("code".into(), J::n(-32601.0)),
                    ("message".into(), J::s(&format!("method not found: {}", method)))]))]));
        }
    };
    id.map(|id| J::Obj(vec![
        ("jsonrpc".into(), J::s("2.0")), ("id".into(), id), ("result".into(), result)]))
}

pub fn serve(brain_path: &str, engine: Engine) -> Result<(), String> {
    let mut b = Brain::open(Some(brain_path))?;
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        let line = line.trim();
        if line.is_empty() { continue; }
        let msg = match json::parse(line.as_bytes()) { Ok(m) => m, Err(_) => continue };
        if let Some(resp) = handle(&mut b, &engine, &msg) {
            let _ = writeln!(stdout, "{}", json::write(&resp));
            let _ = stdout.flush();
        }
    }
    Ok(())
}
