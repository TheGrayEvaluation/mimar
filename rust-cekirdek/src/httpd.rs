//! Kalıcı HTTP servisi — çok-kullanıcı/eşzamanlılık + DAYANIKLILIK (WAL) +
//! çok-instance (okuma replikası). Beyin RAM'de; RwLock ile eşzamanlı okuma,
//! seri yazma. Saf std, sıfır bağımlılık.
//!
//! Dayanıklılık: her yazma önce RAM'e uygulanır + append-only WAL'a (<beyin>.wal)
//! yazılır (ucuz). Arka plan iş parçacığı periyodik SNAPSHOT (tam beyin, atomik)
//! alıp WAL'ı kırpar. Açılışta: snapshot yükle + WAL'ı replay et → sıfır kayıp.
//!
//! Çok-instance: `--replica` modunda yazma reddedilir; servis beyin dosyasının
//! mtime'ını izler, değişince yeniden yükler (yazıcı instance'ın yazdığını görür).

use crate::json::{self, J};
use crate::store::{Brain, Meta};
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Read, Write};
use std::collections::HashMap;
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, SystemTime};

static READS: AtomicU64 = AtomicU64::new(0);
static WRITES: AtomicU64 = AtomicU64::new(0);
static DIRTY: AtomicBool = AtomicBool::new(false);
static START: AtomicU64 = AtomicU64::new(0);   // servis başlangıç epoch (uptime)

struct Ctx {
    brain: RwLock<Brain>,
    wal: Mutex<Option<std::fs::File>>,   // yazıcı: WAL append; replika: None
    path: String,
    replica: bool,
    admin_token: String,                 // boş = auth kapalı; dolu = yazma Bearer ister
    read_token: String,                  // RBAC: dolu = OKUMALAR da Bearer ister (read|admin)
    rate: f64,                           // IP başına saniyedeki istek (0 = kapalı)
    rl: Mutex<HashMap<String, (f64, u128)>>,  // token-bucket: ip → (jeton, son_ms)
}

fn now_ms() -> u128 {
    SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}

/// IP başına token-bucket rate-limit. Dolu kova = rate×2 (patlama payı). Her istek 1 jeton.
fn rate_ok(ctx: &Ctx, ip: &str) -> bool {
    if ctx.rate <= 0.0 { return true; }
    let now = now_ms();
    let mut m = match ctx.rl.lock() { Ok(g) => g, Err(_) => return true };
    let e = m.entry(ip.to_string()).or_insert((ctx.rate, now));
    let dt = (now.saturating_sub(e.1)) as f64 / 1000.0;
    e.0 = (e.0 + dt * ctx.rate).min(ctx.rate * 2.0);   // yeniden doldur, patlama tavanı
    e.1 = now;
    if e.0 >= 1.0 { e.0 -= 1.0; true } else { false }
}

fn s<'a>(j: &'a J, k: &str) -> &'a str { j.get(k).and_then(|v| v.as_str()).unwrap_or("") }

fn meta_from(op: &J) -> Meta {
    let slots: Vec<String> = match op.get("slots") {
        Some(J::Arr(a)) => a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect(),
        _ => s(op, "slots").split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect(),
    };
    Meta {
        collection: { let c = s(op, "collection"); if c.is_empty() { None } else { Some(c.to_string()) } },
        tool: { let t = s(op, "tool"); if t.is_empty() { None } else { Some(t.to_string()) } },
        slots,
        ttl_sec: op.get("ttl").and_then(|v| v.as_f64()),
        volatility: None,
        scope: { let sc = s(op, "scope"); if sc.is_empty() { None } else { Some(sc.to_string()) } },
    }
}

/// Bir yazma op'unu beyne uygula (replay + canlı ortak yol).
fn apply_op(b: &mut Brain, op: &J) {
    match s(op, "op") {
        "teach" => { b.teach(s(op, "prompt"), s(op, "answer")); }
        "note" => { b.note(s(op, "prompt"), s(op, "answer")); }
        "define" => { let m = meta_from(op); b.define(s(op, "query"), s(op, "answer"), &m); }
        "forget" => { b.forget(s(op, "query")); }
        "policy" => {
            let mode = { let m = s(op, "mod"); if m.is_empty() { None } else { Some(m.to_string()) } };
            b.set_policy(mode.as_deref(), op.get("esik").and_then(|v| v.as_f64()),
                         op.get("ogrenme").map(|v| v.as_bool()));
        }
        _ => {}
    }
}

fn wal_append(ctx: &Ctx, op: &J) {
    if let Ok(mut g) = ctx.wal.lock() {
        if let Some(f) = g.as_mut() {
            let _ = writeln!(f, "{}", json::write(op));
            let _ = f.flush();
        }
    }
    DIRTY.store(true, Ordering::Relaxed);
}

/// Sunucu-tarafı TLS seçeneği. `tls` özelliği olmadan yalnız Off çalışır.
pub enum TlsOpt { Off, SelfSigned, Files(String, String) }

pub fn serve_http(brain_path: &str, addr: &str, replica: bool, admin_token: &str) -> Result<(), String> {
    serve_http_rbac(brain_path, addr, replica, admin_token, "", 0.0)
}

/// RBAC'li sürüm: read_token dolu ise okumalar da Bearer (read veya admin) ister.
/// `rate` > 0 ise IP başına saniyedeki istek sınırı (token-bucket, patlama rate×2).
pub fn serve_http_rbac(brain_path: &str, addr: &str, replica: bool,
                       admin_token: &str, read_token: &str, rate: f64) -> Result<(), String> {
    serve_http_full(brain_path, addr, replica, admin_token, read_token, rate, TlsOpt::Off)
}

/// Tam sürüm: TLS dahil. `tls` Off değilse HTTPS (sertifika dosyadan ya da kendinden-imzalı).
pub fn serve_http_full(brain_path: &str, addr: &str, replica: bool, admin_token: &str,
                       read_token: &str, rate: f64, tls: TlsOpt) -> Result<(), String> {
    START.store(now() as u64, Ordering::Relaxed);
    let mut brain = Brain::open(brain_path.into())?;
    let wal_path = format!("{}.wal", brain_path);

    // Açılışta WAL replay (yazıcı; kayıp kurtarma)
    if !replica {
        if let Ok(f) = std::fs::File::open(&wal_path) {
            let mut n = 0;
            for line in BufReader::new(f).lines().flatten() {
                if line.trim().is_empty() { continue; }
                if let Ok(op) = json::parse(line.as_bytes()) { apply_op(&mut brain, &op); n += 1; }
            }
            if n > 0 { eprintln!("WAL replay: {} işlem kurtarıldı", n); let _ = brain.save(); let _ = std::fs::remove_file(&wal_path); }
        }
    }

    let wal = if replica { None } else {
        Some(OpenOptions::new().create(true).append(true).open(&wal_path)
            .map_err(|e| format!("WAL açılamadı: {}", e))?)
    };
    let ctx = Arc::new(Ctx { brain: RwLock::new(brain), wal: Mutex::new(wal),
                             path: brain_path.to_string(), replica,
                             admin_token: admin_token.to_string(),
                             read_token: read_token.to_string(),
                             rate, rl: Mutex::new(HashMap::new()) });

    // Arka plan: yazıcı→periyodik snapshot+WAL kırp; replika→mtime izle+reload
    {
        let ctx = Arc::clone(&ctx);
        let wal_path = wal_path.clone();
        thread::spawn(move || {
            let mut last_mtime = file_mtime(&ctx.path);
            loop {
                thread::sleep(Duration::from_secs(2));
                if ctx.replica {
                    let m = file_mtime(&ctx.path);
                    if m != last_mtime {
                        if let Ok(nb) = Brain::open(ctx.path.as_str().into()) {
                            *ctx.brain.write().unwrap() = nb;
                            last_mtime = m;
                            eprintln!("replika: beyin yeniden yüklendi (mtime değişti)");
                        }
                    }
                } else if DIRTY.swap(false, Ordering::Relaxed) {
                    { let mut b = ctx.brain.write().unwrap(); let _ = b.save(); }
                    // snapshot alındı → WAL'ı kırp
                    if let Ok(mut g) = ctx.wal.lock() {
                        if let Ok(f) = OpenOptions::new().write(true).truncate(true).create(true).open(&wal_path) {
                            *g = Some(f);
                        }
                    }
                }
            }
        });
    }

    let listener = TcpListener::bind(addr).map_err(|e| format!("bind hatası {}: {}", addr, e))?;
    let use_tls = !matches!(tls, TlsOpt::Off);
    if use_tls && !cfg!(feature = "tls") {
        return Err("TLS için `--features tls` ile derleyin".into());
    }
    let proto = if use_tls { "https" } else { "http" };
    eprintln!("mimar HTTP servisi: {}://{}  (RAM beyin · {}{})",
              proto, addr, if replica { "REPLİKA/okuma" } else { "yazıcı+WAL" },
              if replica { "" } else { " · eşzamanlı okuma/seri yazma" });

    #[cfg(feature = "tls")]
    if use_tls {
        let cfg = build_tls(&tls)?;
        for stream in listener.incoming() {
            if let Ok(mut st) = stream {
                let ip = st.peer_addr().map(|a| a.ip().to_string()).unwrap_or_default();
                let ctx = Arc::clone(&ctx);
                let cfg = Arc::clone(&cfg);
                thread::spawn(move || {
                    match rustls::ServerConnection::new(cfg) {
                        Ok(mut conn) => {
                            let tls_stream = rustls::Stream::new(&mut conn, &mut st);
                            let _ = handle(tls_stream, ip, ctx);
                        }
                        Err(_) => {}
                    }
                });
            }
        }
        return Ok(());
    }

    for stream in listener.incoming() {
        if let Ok(st) = stream {
            let ip = st.peer_addr().map(|a| a.ip().to_string()).unwrap_or_default();
            let ctx = Arc::clone(&ctx);
            thread::spawn(move || { let _ = handle(st, ip, ctx); });
        }
    }
    Ok(())
}

/// TLS yapılandırması (sertifika+anahtar) kur — dosyadan ya da kendinden-imzalı.
#[cfg(feature = "tls")]
fn build_tls(tls: &TlsOpt) -> Result<Arc<rustls::ServerConfig>, String> {
    use rustls::pki_types::{CertificateDer, PrivateKeyDer};
    let (certs, key): (Vec<CertificateDer<'static>>, PrivateKeyDer<'static>) = match tls {
        TlsOpt::SelfSigned => {
            // Kendinden-imzalı sertifika üret (localhost) — hızlı başlangıç/geliştirme.
            let c = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
                .map_err(|e| format!("sertifika üretilemedi: {}", e))?;
            let cert = CertificateDer::from(c.cert.der().to_vec());
            let key = PrivateKeyDer::try_from(c.key_pair.serialize_der())
                .map_err(|e| format!("anahtar: {}", e))?;
            eprintln!("TLS: kendinden-imzalı sertifika üretildi (CN=localhost · yalnız geliştirme)");
            (vec![cert], key)
        }
        TlsOpt::Files(cert_path, key_path) => {
            let cf = std::fs::read(cert_path).map_err(|e| format!("sertifika okunamadı: {}", e))?;
            let kf = std::fs::read(key_path).map_err(|e| format!("anahtar okunamadı: {}", e))?;
            let certs: Vec<CertificateDer<'static>> =
                rustls_pemfile::certs(&mut &cf[..]).collect::<Result<_, _>>()
                    .map_err(|e| format!("sertifika PEM: {}", e))?;
            let key = rustls_pemfile::private_key(&mut &kf[..])
                .map_err(|e| format!("anahtar PEM: {}", e))?
                .ok_or("anahtar bulunamadı (PEM)")?;
            (certs, key)
        }
        TlsOpt::Off => return Err("TLS kapalı".into()),
    };
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    rustls::ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|e| format!("TLS protokol: {}", e))?
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map(Arc::new)
        .map_err(|e| format!("TLS yapılandırma: {}", e))
}

#[cfg(all(test, feature = "tls"))]
mod tls_tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpStream;

    #[derive(Debug)]
    struct NoVerify(Arc<rustls::crypto::CryptoProvider>);
    impl rustls::client::danger::ServerCertVerifier for NoVerify {
        fn verify_server_cert(&self, _e: &rustls::pki_types::CertificateDer<'_>,
            _i: &[rustls::pki_types::CertificateDer<'_>], _n: &rustls::pki_types::ServerName<'_>,
            _o: &[u8], _t: rustls::pki_types::UnixTime)
            -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
            Ok(rustls::client::danger::ServerCertVerified::assertion())
        }
        fn verify_tls12_signature(&self, _m: &[u8], _c: &rustls::pki_types::CertificateDer<'_>,
            _d: &rustls::DigitallySignedStruct)
            -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
            Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
        }
        fn verify_tls13_signature(&self, _m: &[u8], _c: &rustls::pki_types::CertificateDer<'_>,
            _d: &rustls::DigitallySignedStruct)
            -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
            Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
        }
        fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
            self.0.signature_verification_algorithms.supported_schemes()
        }
    }

    #[test]
    fn https_self_signed_end_to_end() {
        let tmp = std::env::temp_dir().join("mimar_tls_test.mimar");
        let _ = std::fs::remove_file(&tmp);
        { let mut b = Brain::open(Some(tmp.to_str().unwrap())).unwrap();
          b.teach("selam", "merhaba"); b.save().unwrap(); }
        let path = tmp.to_str().unwrap().to_string();
        let addr = "127.0.0.1:17788";
        thread::spawn(move || {
            let _ = serve_http_full(&path, addr, false, "", "", 0.0, TlsOpt::SelfSigned);
        });
        thread::sleep(Duration::from_millis(500));

        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let cfg = rustls::ClientConfig::builder_with_provider(provider.clone())
            .with_safe_default_protocol_versions().unwrap()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(NoVerify(provider)))
            .with_no_client_auth();
        let name = rustls::pki_types::ServerName::try_from("localhost").unwrap();
        let mut conn = rustls::ClientConnection::new(Arc::new(cfg), name).unwrap();
        let mut sock = TcpStream::connect(addr).unwrap();
        let mut tls = rustls::Stream::new(&mut conn, &mut sock);
        tls.write_all(b"GET /saglik HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n").unwrap();
        let mut buf = Vec::new();
        let _ = tls.read_to_end(&mut buf);
        let resp = String::from_utf8_lossy(&buf);
        assert!(resp.contains("\"ok\":true"), "HTTPS yanıtı beklenen değil: {}", resp);
        let _ = std::fs::remove_file(&tmp);
    }
}

fn now() -> u128 {
    SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as u128).unwrap_or(0)
}

fn file_mtime(p: &str) -> u128 {
    std::fs::metadata(p).ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos()).unwrap_or(0)
}

struct Req { method: String, target: String, body: Vec<u8>, auth: String }

fn read_req<R: Read>(stream: &mut R) -> Option<Req> {
    let mut r = BufReader::new(stream);
    let mut first = String::new();
    if r.read_line(&mut first).ok()? == 0 { return None; }
    let mut parts = first.split_whitespace();
    let method = parts.next()?.to_string();
    let target = parts.next()?.to_string();
    let mut clen = 0usize;
    let mut auth = String::new();
    loop {
        let mut line = String::new();
        if r.read_line(&mut line).ok()? == 0 { break; }
        let t = line.trim_end();
        if t.is_empty() { break; }
        let tl = t.to_ascii_lowercase();
        if let Some(v) = tl.strip_prefix("content-length:") {
            clen = v.trim().parse().unwrap_or(0);
        } else if tl.starts_with("authorization:") {
            if let Some(val) = t.splitn(2, ':').nth(1) {
                let val = val.trim();
                let tok = val.strip_prefix("Bearer ")
                    .or_else(|| val.strip_prefix("bearer ")).unwrap_or(val);
                auth = tok.trim().to_string();
            }
        }
    }
    let mut body = vec![0u8; clen];
    if clen > 0 { r.read_exact(&mut body).ok()?; }
    Some(Req { method, target, body, auth })
}

fn send<W: Write>(stream: &mut W, code: &str, body: &str) {
    let resp = format!(
        "HTTP/1.1 {}\r\nContent-Type: application/json; charset=utf-8\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        code, body.as_bytes().len(), body);
    let _ = stream.write_all(resp.as_bytes());
    let _ = stream.flush();
}

fn send_html<W: Write>(stream: &mut W, body: &str) {
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.as_bytes().len(), body);
    let _ = stream.write_all(resp.as_bytes());
    let _ = stream.flush();
}

/// Gömülü canlı panel — /metrics'i periyodik çekip gösterir (sıfır bağımlılık, inline).
const DASHBOARD_HTML: &str = r#"<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Mimar · Panel</title>
<style>:root{color-scheme:dark light}body{margin:0;font:16px/1.5 system-ui,sans-serif;
background:#0b1118;color:#e8eef6}header{padding:22px 26px;border-bottom:1px solid #223246}
h1{margin:0;font-size:22px}h1 b{color:#49c7ff}.wrap{padding:26px;display:grid;
grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;max-width:1000px}
.c{background:#121d2c;border:1px solid #223246;border-radius:14px;padding:20px}
.k{color:#92a3b8;font-size:13px;text-transform:uppercase;letter-spacing:1px}
.v{font-size:40px;font-weight:700;font-family:ui-monospace,monospace;margin-top:6px;color:#41e08a}
.s{padding:0 26px 26px;color:#5d6e85;font:13px ui-monospace,monospace}
@media(prefers-color-scheme:light){body{background:#f6f8fb;color:#0b1118}.c{background:#fff;border-color:#dce3ec}}
</style></head><body>
<header><h1><b>Mimar</b> · canlı panel</h1></header>
<div class="wrap">
<div class="c"><div class="k">Çalışma süresi</div><div class="v" id="up">—</div></div>
<div class="c"><div class="k">Niyet (bilgi)</div><div class="v" id="ni">—</div></div>
<div class="c"><div class="k">Okuma</div><div class="v" id="rd">—</div></div>
<div class="c"><div class="k">Yazma</div><div class="v" id="wr">—</div></div>
<div class="c"><div class="k">Rol</div><div class="v" id="rp" style="font-size:24px">—</div></div>
</div>
<div class="s" id="st">bağlanılıyor…</div>
<script>
function f(s){var h=Math.floor(s/3600),m=Math.floor(s%3600/60),x=s%60;return h+"s "+m+"d "+x+"sn";}
async function tick(){try{var r=await fetch("/metrics",{cache:"no-store"});var j=await r.json();
up.textContent=f(j.uptime_sn);ni.textContent=j.niyet;rd.textContent=j.reads;wr.textContent=j.writes;
rp.textContent=(j.replica?"replika":"yazıcı")+(j.rbac_okuma?" · RBAC":"");
st.textContent="son güncelleme "+new Date().toLocaleTimeString()+" · 2 sn'de bir yenilenir";
}catch(e){st.textContent="metrics alınamadı: "+e+(location.search?"":" (RBAC açıksa token gerekir)");}}
tick();setInterval(tick,2000);
</script></body></html>"#;

fn handle<S: Read + Write>(mut stream: S, ip: String, ctx: Arc<Ctx>) -> std::io::Result<()> {
    let req = match read_req(&mut stream) { Some(r) => r, None => return Ok(()) };
    let body: J = json::parse(&req.body).unwrap_or(J::Obj(vec![]));
    let t = req.target.as_str();

    // ---- RATE-LIMIT (IP başına token-bucket) ----
    if !rate_ok(&ctx, &ip) {
        return Ok(send(&mut stream, "429 Too Many Requests",
            "{\"hata\":\"hız sınırı aşıldı (rate limit)\"}"));
    }

    // ---- DASHBOARD: tarayıcıda canlı panel (/ veya /dashboard) ----
    if req.method == "GET" && (t == "/" || t == "/dashboard") {
        return Ok(send_html(&mut stream, DASHBOARD_HTML));
    }

    // ---- RBAC: okuma-token ayarlıysa TÜM istekler read VEYA admin token ister ----
    if !ctx.read_token.is_empty() && req.auth != ctx.read_token && req.auth != ctx.admin_token {
        return Ok(send(&mut stream, "401 Unauthorized",
            "{\"hata\":\"RBAC: okuma için geçerli token gerekli (Authorization: Bearer ...)\"}"));
    }

    // ---- OKUMA (eşzamanlı) ----
    if req.method == "GET" && t == "/saglik" {
        return Ok(send(&mut stream, "200 OK",
            &format!("{{\"ok\":true,\"replica\":{},\"reads\":{},\"writes\":{}}}",
                ctx.replica, READS.load(Ordering::Relaxed), WRITES.load(Ordering::Relaxed))));
    }
    // ---- GÖZLEMLENEBİLİRLİK: /metrics (uptime, niyet, okuma/yazma, replika) ----
    if req.method == "GET" && t == "/metrics" {
        let intents = { ctx.brain.read().unwrap().cache_len() };
        let up = now().saturating_sub(START.load(Ordering::Relaxed) as u128);
        return Ok(send(&mut stream, "200 OK", &format!(
            "{{\"uptime_sn\":{},\"niyet\":{},\"reads\":{},\"writes\":{},\"replica\":{},\"rbac_okuma\":{}}}",
            up, intents, READS.load(Ordering::Relaxed), WRITES.load(Ordering::Relaxed),
            ctx.replica, !ctx.read_token.is_empty())));
    }
    if req.method == "GET" && t == "/stats" {
        let b = ctx.brain.read().unwrap();
        return Ok(send(&mut stream, "200 OK",
            &format!("{{\"niyet\":{},\"reads\":{},\"writes\":{}}}",
                b.cache_len(), READS.load(Ordering::Relaxed), WRITES.load(Ordering::Relaxed))));
    }
    if req.method == "GET" && t == "/list" {
        let b = ctx.brain.read().unwrap();
        READS.fetch_add(1, Ordering::Relaxed);
        return Ok(send(&mut stream, "200 OK", &json::write(&b.intents_json())));
    }
    if req.method == "POST" && t == "/resolve" {
        let q = s(&body, "query").to_string();
        let user = s(&body, "user");
        let uopt = if user.is_empty() { None } else { Some(user) };
        let d = { let b = ctx.brain.read().unwrap(); b.resolve_scoped(&q, uopt) };
        READS.fetch_add(1, Ordering::Relaxed);
        return Ok(send(&mut stream, "200 OK", &json::write(&d.to_json())));
    }
    if req.method == "POST" && t == "/redact" {
        let masked = crate::store::redact(s(&body, "text"));
        return Ok(send(&mut stream, "200 OK",
            &json::write(&J::Obj(vec![("redacted".into(), J::s(&masked))]))));
    }

    // ---- YAZMA (admin) ----
    if ctx.replica {
        return Ok(send(&mut stream, "403 Forbidden",
            "{\"hata\":\"replika salt-okunur; yazma yazıcı instance'a\"}"));
    }
    // Auth: token ayarlıysa yazma işlemleri Bearer token ister (okumalar serbest).
    if !ctx.admin_token.is_empty() && req.auth != ctx.admin_token {
        return Ok(send(&mut stream, "401 Unauthorized",
            "{\"hata\":\"geçersiz veya eksik admin token (Authorization: Bearer ...)\"}"));
    }
    // TOPLU yazma: tek bağlantı + tek kilit + tek flush ile binlerce op (hızlı seeding).
    if req.method == "POST" && t == "/batch" {
        let ops: Vec<J> = match body.get("ops") { Some(J::Arr(a)) => a.clone(), _ => Vec::new() };
        {
            let mut b = ctx.brain.write().unwrap();
            for op in &ops { apply_op(&mut b, op); }
        }
        if let Ok(mut g) = ctx.wal.lock() {
            if let Some(f) = g.as_mut() {
                for op in &ops { let _ = writeln!(f, "{}", json::write(op)); }
                let _ = f.flush();
            }
        }
        DIRTY.store(true, Ordering::Relaxed);
        WRITES.fetch_add(ops.len() as u64, Ordering::Relaxed);
        return Ok(send(&mut stream, "200 OK",
            &format!("{{\"ok\":true,\"applied\":{}}}", ops.len())));
    }
    let (ok_msg, op): (String, Option<J>) = match (req.method.as_str(), t) {
        ("POST", "/teach") => {
            let o = { let mut b = ctx.brain.write().unwrap(); b.teach(s(&body, "prompt"), s(&body, "answer")) };
            (format!("{{\"ok\":true,\"sonuc\":\"{}\"}}", o.label()),
             Some(J::Obj(vec![("op".into(), J::s("teach")), ("prompt".into(), J::s(s(&body,"prompt"))),
                              ("answer".into(), J::s(s(&body,"answer")))])))
        }
        ("POST", "/note") => {
            { let mut b = ctx.brain.write().unwrap(); b.note(s(&body, "prompt"), s(&body, "answer")); }
            ("{\"ok\":true,\"sonuc\":\"model notu\"}".into(),
             Some(J::Obj(vec![("op".into(), J::s("note")), ("prompt".into(), J::s(s(&body,"prompt"))),
                              ("answer".into(), J::s(s(&body,"answer")))])))
        }
        ("POST", "/define") => {
            let m = meta_from(&body);
            let o = { let mut b = ctx.brain.write().unwrap(); b.define(s(&body, "query"), s(&body, "answer"), &m) };
            let mut op = body.clone();
            if let J::Obj(ref mut e) = op { e.insert(0, ("op".into(), J::s("define"))); }
            (format!("{{\"ok\":true,\"sonuc\":\"{}\"}}", o.label()), Some(op))
        }
        ("POST", "/policy") => {
            let mode = { let m = s(&body, "mod"); if m.is_empty() { None } else { Some(m.to_string()) } };
            { let mut b = ctx.brain.write().unwrap();
              b.set_policy(mode.as_deref(), body.get("esik").and_then(|v| v.as_f64()),
                           body.get("ogrenme").map(|v| v.as_bool())); }
            let mut op = body.clone();
            if let J::Obj(ref mut e) = op { e.insert(0, ("op".into(), J::s("policy"))); }
            ("{\"ok\":true}".into(), Some(op))
        }
        ("POST", "/forget") => {
            let removed = { let mut b = ctx.brain.write().unwrap(); b.forget(s(&body, "query")) };
            (format!("{{\"ok\":{}}}", removed),
             Some(J::Obj(vec![("op".into(), J::s("forget")), ("query".into(), J::s(s(&body,"query")))])))
        }
        _ => return Ok(send(&mut stream, "404 Not Found", "{\"hata\":\"yok\"}")),
    };
    if let Some(op) = op { wal_append(&ctx, &op); }   // dayanıklılık: WAL + dirty
    WRITES.fetch_add(1, Ordering::Relaxed);
    Ok(send(&mut stream, "200 OK", &ok_msg))
}
