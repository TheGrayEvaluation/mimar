//! Mimar CLI — native ana motor girişi (`mimar`), SQLite'ın `sqlite3`'ü gibi.
//! Her yerden çağrılır; .mimar beyni oluşturur/öğretir/sorar; model için yerel
//! Ollama/LM Studio'yu çağırır.

use mimar_core::embed;
use mimar_core::engine::Engine;
use mimar_core::store::{merge_brains, Brain, Meta};
use mimar_core::json::{self};
use std::env;
use std::process;

fn opt(args: &[String], name: &str, default: &str) -> String {
    for i in 0..args.len() {
        if args[i] == name && i + 1 < args.len() {
            return args[i + 1].clone();
        }
    }
    default.to_string()
}

fn has(args: &[String], name: &str) -> bool { args.iter().any(|a| a == name) }

fn opt_env(args: &[String], name: &str, env: &str) -> String {
    let v = opt(args, name, "");
    if v.is_empty() { env::var(env).unwrap_or_default() } else { v }
}

/// Motor seç: --provider openrouter | --base-url <url> (bulut, TLS) | --lmstudio | (vars.) Ollama.
/// Anahtar: --key veya ortam değişkeni (OPENROUTER_API_KEY / OPENAI_API_KEY).
fn pick_engine(args: &[String], model: &str) -> Engine {
    let provider = opt(args, "--provider", "");
    let base = opt(args, "--base-url", "");
    if provider == "openrouter" {
        Engine::openrouter(model, &opt_env(args, "--key", "OPENROUTER_API_KEY"))
    } else if !base.is_empty() {
        Engine::cloud(&base, model, &opt_env(args, "--key", "OPENAI_API_KEY"))
    } else if has(args, "--lmstudio") {
        Engine::openai_local(model, 1234)
    } else {
        Engine::ollama(model)
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Mimar — yapay zekânın SQLite'ı (native çekirdek)\n\
                   kullanım:\n\
                   \tmimar new   <beyin.mimar> [--role \"...\"] [--guard \"...\"]\n\
                   \tmimar teach <beyin.mimar> \"soru\" \"cevap\"  (varsa üstüne yazar = upsert)\n\
                   \tmimar forget <beyin.mimar> \"soru\"         (niyeti sil)\n\
                   \tmimar feedback <beyin.mimar> \"soru\" good|bad\n\
                   \tmimar ask   <beyin.mimar> \"soru\" [--model llama3.2:3b] [--lmstudio]\n\
                   \tmimar recall <beyin.mimar> \"soru\"   (sadece hafıza, basit)\n\
                   \tmimar resolve <beyin.mimar> \"soru\"  (tipli direktif JSON: fact/rule/tool/stale/miss)\n\
                   \tmimar define <beyin.mimar> \"soru\" --tip kural|arac --slots saat --tool ad --ttl 0\n\
                   \tmimar policy <beyin.mimar> [--mod karma|sadece_yz|sadece_mimar] [--esik 0.65] [--ogrenme acik|kapali]\n\
                   \tmimar stats <beyin.mimar>\n\
                   \tmimar embed \"metin\"   (Python paritesi için)\n\
                   \tmimar key   \"metin\"");
        process::exit(2);
    }
    let cmd = args[1].as_str();

    // Python paritesi yardımcıları (beyin gerektirmez)
    if cmd == "embed" {
        let v = embed::embed(&args[2]);
        println!("{}", v.iter().map(|x| format!("{:.9}", x)).collect::<Vec<_>>().join(","));
        return;
    }
    if cmd == "key" {
        println!("{}", embed::key(&args[2]));
        return;
    }

    let path = args.get(2).cloned().unwrap_or_default();

    let result: Result<(), String> = (|| {
        match cmd {
            "new" => {
                let mut b = Brain::open(Some(&path))?;
                let role = opt(&args, "--role", "");
                if !role.is_empty() { b.set_role(&role); }
                let guard = opt(&args, "--guard", "");
                if !guard.is_empty() { b.add_guard(&guard); }
                // --embedder ollama:<model> → opt-in nöral embedder (çapraz-dil/eş-anlam)
                let emb = opt(&args, "--embedder", "");
                if !emb.is_empty() { b.set_embedder(mimar_core::embedder::make_embedder(&emb)); }
                b.save()?;
                println!("oluşturuldu: {} (feature={})", path, b.feature());
            }
            "reembed" => {
                let mut b = Brain::open(Some(&path))?;
                let emb = opt(&args, "--embedder", &b.embedder_id());
                let n = b.set_embedder(mimar_core::embedder::make_embedder(&emb));
                b.save()?;
                println!("yeniden gömüldü: {} niyet (feature={})", n, b.embedder_id());
            }
            "calibrate" => {
                let mut b = Brain::open(Some(&path))?;
                let (bar, n, maxc) = b.calibrate();
                b.save()?;
                println!("kalibre edildi: {} niyet · güven eşiği={:.3} \
                          (en yüksek farklı-cevap benzerliği={:.3})", n, bar, maxc);
            }
            "teach" => {
                let mut b = Brain::open(Some(&path))?;
                let outcome = b.teach(&args[3], &args[4]);
                b.save()?;
                println!("öğretildi [{}]: «{}» -> {}", outcome.label(), args[3], args[4]);
            }
            "recall" => {
                // Sadece hafıza — model çağrılmaz (geriye-uyum; basit isabet/ışka).
                let b = Brain::open(Some(&path))?;
                match b.recall(&args[3]) {
                    Some(h) => {
                        let src = if h.learned { "rule" } else { "cache" };
                        println!("[{} sim={:.3}] {}", src, h.sim, h.answer);
                    }
                    None => println!("[miss]"),
                }
            }
            "resolve" => {
                // YENİ pipeline: tipli direktif (JSON). Politika+koleksiyon+TTL+slot uygular.
                let b = Brain::open(Some(&path))?;
                let d = b.resolve(&args[3]);
                println!("{}", json::write(&d.to_json()));
            }
            "define" => {
                // Tipli niyet tanımla: mimar define <beyin> "soru" --tip arac --slots tarih --tool musaitlik --ttl 0 [--answer "..."]
                let mut b = Brain::open(Some(&path))?;
                let tip = opt(&args, "--tip", "kesin");
                let tool = opt(&args, "--tool", "");
                let ttl = opt(&args, "--ttl", "");
                let slots_s = opt(&args, "--slots", "");
                let answer = opt(&args, "--answer", "");
                let slots: Vec<String> = slots_s.split(',')
                    .map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
                let meta = Meta {
                    collection: Some(tip.clone()),
                    tool: if tool.is_empty() { None } else { Some(tool) },
                    slots,
                    ttl_sec: ttl.parse::<f64>().ok(),
                    volatility: Some(if tip == "arac" || tip == "eylem" { "anlik".into() }
                                     else if tip == "kural" { "statik".into() } else { "statik".into() }),
                    scope: { let s = opt(&args, "--scope", ""); if s.is_empty() { None } else { Some(s) } },
                };
                let outcome = b.define(&args[3], &answer, &meta);
                b.save()?;
                println!("tanımlandı [{}] tip={}: «{}»", outcome.label(), tip, args[3]);
            }
            "policy" => {
                let mut b = Brain::open(Some(&path))?;
                let set = has(&args, "--mod") || has(&args, "--esik") || has(&args, "--ogrenme");
                if set {
                    let mode = if has(&args, "--mod") { Some(opt(&args, "--mod", "karma")) } else { None };
                    let esik = if has(&args, "--esik") { opt(&args, "--esik", "0.65").parse::<f64>().ok() } else { None };
                    let ogr = if has(&args, "--ogrenme") {
                        Some(matches!(opt(&args, "--ogrenme", "acik").as_str(), "acik" | "açık" | "on" | "true"))
                    } else { None };
                    b.set_policy(mode.as_deref(), esik, ogr);
                    b.save()?;
                }
                let p = b.policy();
                println!("politika: mod={} esik={:.2} ogrenme={}", p.mode, p.esik, p.ogrenme);
            }
            "forget" => {
                let mut b = Brain::open(Some(&path))?;
                let removed = b.forget(&args[3]);
                b.save()?;
                println!("{}", if removed {
                    format!("unutuldu: «{}»", args[3])
                } else {
                    format!("bulunamadı: «{}»", args[3])
                });
            }
            "ask" => {
                let mut b = Brain::open(Some(&path))?;
                let model = opt(&args, "--model", "llama3.2:3b");
                let engine = pick_engine(&args, &model);
                let risk_high = opt(&args, "--risk", "normal") == "high";
                let t0 = std::time::Instant::now();
                let (ans, src, sim) = b.ask(&args[3], &engine, risk_high)?;
                let ms = t0.elapsed().as_millis();
                b.save()?;
                println!("[{} sim={:.3} {}ms] {}", src, sim, ms, ans);
            }
            "explain" => {
                let b = Brain::open(Some(&path))?;
                let risk_high = opt(&args, "--risk", "normal") == "high";
                println!("{}", b.explain(&args[3], risk_high));
            }
            "feedback" => {
                let mut b = Brain::open(Some(&path))?;
                let good = args.get(4).map(|s| s == "good" || s == "iyi").unwrap_or(false);
                let msg = b.feedback(&args[3], good);
                b.save()?;
                println!("{}", msg);
            }
            "sign" => {
                let mut b = Brain::open(Some(&path))?;
                let author = opt(&args, "--author", "");
                let msg = b.sign(&args[3], &author);
                b.save()?;
                println!("{}", msg);
            }
            "verify" => {
                let b = Brain::open(Some(&path))?;
                let (ok, msg) = b.verify(&args[3]);
                println!("{} — {}", if ok { "GEÇERLİ ✓" } else { "GEÇERSİZ ✗" }, msg);
                if !ok { process::exit(1); }
            }
            "merge" => {
                let out = &path;
                let mut srcs = Vec::new();
                for p in &args[3..] {
                    srcs.push(Brain::open(Some(p))?);
                }
                if srcs.is_empty() { return Err("en az bir kaynak beyin ver".into()); }
                let raw = merge_brains(&srcs);
                let n = match raw.get("cache") { Some(json::J::Obj(c)) => c.len(), _ => 0 };
                mimar_core::store::save_raw(out, &raw)?;
                println!("birleştirildi → {} ({} niyet, {} kaynak, provenanslı)",
                         out, n, srcs.len());
            }
            "note" => {
                let mut b = Brain::open(Some(&path))?;
                b.note(&args[3], &args[4]);
                b.save()?;
            }
            "consolidate" => {
                let mut b = Brain::open(Some(&path))?;
                let max: usize = opt(&args, "--max-intents", "0").parse().unwrap_or(0);
                let (before, am, after, ev) = b.consolidate(max);
                b.save()?;
                println!("niyet: {} → birleşme {} → son {} (birleşen {}, budanan {})",
                         before, am, after, before - am, ev);
            }
            "fit" => {
                let mut b = Brain::open(Some(&path))?;
                let epochs: usize = opt(&args, "--epochs", "40").parse().unwrap_or(40);
                let n = b.fit(epochs);
                b.save()?;
                println!("ML eğitildi: {} örnek × {} epoch, {} sınıf",
                         n, epochs, b.ml.classes.len());
            }
            // Davranış damıtma: bir modelin davranışını yakala (teach) → .mimar'a damıt
            "capture" => {
                let mut b = Brain::open(Some(&path))?;
                let model = opt(&args, "--model", "llama3.2:3b");
                let engine = pick_engine(&args, &model);
                let system = opt(&args, "--system", "");
                let pfile = opt(&args, "--prompts", "");
                if pfile.is_empty() { return Err("--prompts <dosya> (satır başı istem) gerekli".into()); }
                let text = std::fs::read_to_string(&pfile).map_err(|e| e.to_string())?;
                let prompts: Vec<String> = text.lines().map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty()).collect();
                let pairs = mimar_core::distill::capture(&mut b, &engine, &prompts, &system)?;
                let out = format!("{}.cases.tsv", path);
                let dump: String = pairs.iter().map(|(p, a)|
                    format!("{}\t{}", p.replace(['\t', '\n'], " "), a.replace(['\t', '\n'], " ")))
                    .collect::<Vec<_>>().join("\n");
                std::fs::write(&out, dump).map_err(|e| e.to_string())?;
                println!("yakalandı: {} davranış çifti ({} modelinden) → beyin {} · eval {}",
                         pairs.len(), model, path, out);
            }
            // Sadakat ölçümü: B+beyin vs B-yalnız (damıtılan davranış taşındı mı?)
            "fidelity" => {
                let mut b = Brain::open(Some(&path))?;
                let model = opt(&args, "--model", "llama3.2:3b");
                let engine = pick_engine(&args, &model);
                let system = opt(&args, "--system", "");
                let thresh: f64 = opt(&args, "--thresh", "0.9").parse().unwrap_or(0.9);
                let cfile = opt(&args, "--cases", "");
                if cfile.is_empty() { return Err("--cases <dosya> (girdi<TAB>beklenen) gerekli".into()); }
                let text = std::fs::read_to_string(&cfile).map_err(|e| e.to_string())?;
                let cases: Vec<(String, String)> = text.lines().filter_map(|l| {
                    let mut it = l.splitn(2, '\t');
                    let i = it.next()?.trim();
                    let e = it.next().unwrap_or("").trim();
                    if i.is_empty() { None } else { Some((i.to_string(), e.to_string())) }
                }).collect();
                let r = mimar_core::distill::fidelity(&mut b, &engine, &cases, thresh)?;
                println!("B+beyin sadakati : {:.0}% ({}/{})  [model: {}]",
                         r.oran * 100.0, r.uyan, r.toplam, model);
                if has(&args, "--baseline") {
                    let rb = mimar_core::distill::raw_fidelity(&engine, &system, &cases, thresh)?;
                    println!("B-yalnız (taban) : {:.0}% ({}/{})", rb.oran * 100.0, rb.uyan, rb.toplam);
                    println!("DAMITILAN DAVRANIŞ KAZANCI: +{:.0} puan", (r.oran - rb.oran) * 100.0);
                }
                if has(&args, "--detay") {
                    for (i, ok, s, o) in &r.detay {
                        println!("  [{}] sim={:.2} {} → {}", if *ok { "✓" } else { "✗" }, s,
                                 i.chars().take(40).collect::<String>(),
                                 o.chars().take(70).collect::<String>());
                    }
                }
            }
            // Güvenlik taraması: prompt-injection/jailbreak/zehirleme sezgisel tespiti
            "scan" => {
                let tags = mimar_core::safety::scan_injection(&args[2]);
                if tags.is_empty() { println!("temiz (risk işareti yok)"); }
                else { println!("RİSK: {}", tags.join(", ")); process::exit(3); }
            }
            // At-rest şifreleme: .mimar dosyasını parolayla şifrele/çöz (encrypt-then-MAC)
            "encrypt" => {
                let outp = opt(&args, "--out", &format!("{}.enc", path));
                let pass = opt_env(&args, "--pass", "MIMAR_PASS");
                if pass.is_empty() { return Err("--pass <parola> veya MIMAR_PASS gerekli".into()); }
                let data = std::fs::read(&path).map_err(|e| e.to_string())?;
                let enc = mimar_core::crypto::encrypt(&data, &pass);
                std::fs::write(&outp, &enc).map_err(|e| e.to_string())?;
                println!("şifrelendi → {} ({} bayt, encrypt-then-MAC)", outp, enc.len());
            }
            "decrypt" => {
                let outp = opt(&args, "--out", &path.trim_end_matches(".enc").to_string());
                let pass = opt_env(&args, "--pass", "MIMAR_PASS");
                if pass.is_empty() { return Err("--pass <parola> veya MIMAR_PASS gerekli".into()); }
                let data = std::fs::read(&path).map_err(|e| e.to_string())?;
                let dec = mimar_core::crypto::decrypt(&data, &pass)?;
                std::fs::write(&outp, &dec).map_err(|e| e.to_string())?;
                println!("çözüldü → {} ({} bayt)", outp, dec.len());
            }
            // Anlık görüntü (point-in-time): beyni zaman damgalı yedeğe kopyala
            "snapshot" => {
                let dir = format!("{}.snapshots", path);
                std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
                let dst = format!("{}/{}.mimar", dir, ts);
                std::fs::copy(&path, &dst).map_err(|e| e.to_string())?;
                println!("anlık görüntü alındı → {}", dst);
            }
            // Geri alma (rollback): --list listeler; --to <dosya> ya da en son yedeğe döner
            "rollback" => {
                let dir = format!("{}.snapshots", path);
                let mut snaps: Vec<String> = std::fs::read_dir(&dir).map(|rd| rd.flatten()
                    .map(|e| e.path().to_string_lossy().to_string())
                    .filter(|p| p.ends_with(".mimar")).collect()).unwrap_or_default();
                snaps.sort();
                if has(&args, "--list") {
                    if snaps.is_empty() { println!("anlık görüntü yok ({})", dir); }
                    else { println!("anlık görüntüler (eskiden yeniye):"); for s in &snaps { println!("  {}", s); } }
                } else {
                    let to = opt(&args, "--to", "");
                    let src = if !to.is_empty() { to }
                        else { snaps.last().cloned().ok_or("geri alınacak anlık görüntü yok")? };
                    // güvenlik: mevcut hâli geri-alma öncesi yedekle
                    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis()).unwrap_or(0);
                    std::fs::create_dir_all(&dir).ok();
                    let _ = std::fs::copy(&path, &format!("{}/{}-rollback-oncesi.mimar", dir, ts));
                    std::fs::copy(&src, &path).map_err(|e| e.to_string())?;
                    // doğrula: geri alınan beyin açılıyor mu
                    let n = Brain::open(Some(&path))?.cache_len();
                    println!("geri alındı: {} → {} ({} niyet; önceki hâl yedeklendi)", src, path, n);
                }
            }
            // Gömülü için DONDUR: beyni int8 pakete nicemle (8× küçük, MCU/WASM)
            "freeze" => {
                let b = Brain::open(Some(&path))?;
                let out = opt(&args, "--out", &format!("{}.mqz", path));
                let pack = b.freeze();
                let bytes = pack.to_bytes();
                std::fs::write(&out, &bytes).map_err(|e| e.to_string())?;
                let orig = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                println!("donduruldu: {} niyet → {} ({} bayt; kaynak {} bayt)",
                         pack.len(), out, bytes.len(), orig);
            }
            // Dondurulmuş paketi yükle + sor (no_std-hazır çekirdek; model YOK)
            "ask-frozen" => {
                let data = std::fs::read(&path).map_err(|e| e.to_string())?;
                let pack = mimar_core::quant::QuantPack::from_bytes(&data)
                    .ok_or("geçersiz .mqz paketi")?;
                let emb = opt(&args, "--embedder", "hash-256");
                let embedder = mimar_core::embedder::make_embedder(&emb);
                let qv = embedder.embed(&args[3]);
                match pack.nearest(&qv) {
                    Some((i, sim)) if sim >= 0.55 =>
                        println!("[frozen sim={:.3}] {}", sim, pack.answers[i]),
                    Some((_, sim)) => println!("[frozen miss sim={:.3}]", sim),
                    None => println!("[boş paket]"),
                }
            }
            "route" => {
                let b = Brain::open(Some(&path))?;
                let (cos, ml) = b.route(&args[3]);
                match cos {
                    Some(h) => println!("cosine : «{}» sim={:.3} (verified={} blocked={})",
                                        h.answer, h.sim, h.verified, h.blocked),
                    None => println!("cosine : (niyet yok)"),
                }
                match ml {
                    Some((q, p)) => println!("ML     : «{}» olasılık={:.3}", q, p),
                    None => println!("ML     : (sınıf yok)"),
                }
            }
            "serve-http" => {
                // Kalıcı HTTP servisi (çok-kullanıcı): beyin RAM'de, eşzamanlı okuma/seri yazma + WAL.
                // --replica: salt-okunur okuma replikası (mtime izler, yazıcıdan reload).
                let addr = opt(&args, "--addr", "127.0.0.1:8899");
                let replica = has(&args, "--replica");
                // --token: ayarlıysa YAZMA uçları Bearer token ister (uzak/n8n güvenliği).
                let token = opt_env(&args, "--token", "MIMAR_ADMIN_TOKEN");
                // --read-token: RBAC — ayarlıysa OKUMALAR da token ister (read veya admin).
                let read_token = opt_env(&args, "--read-token", "MIMAR_READ_TOKEN");
                // --rate: IP başına saniyedeki istek sınırı (token-bucket; 0=kapalı).
                let rate: f64 = opt(&args, "--rate", "0").parse().unwrap_or(0.0);
                // TLS: --tls-self-signed (geliştirme) ya da --tls-cert <pem> --tls-key <pem>.
                let tls = if has(&args, "--tls-self-signed") {
                    mimar_core::httpd::TlsOpt::SelfSigned
                } else {
                    let c = opt(&args, "--tls-cert", ""); let k = opt(&args, "--tls-key", "");
                    if !c.is_empty() && !k.is_empty() { mimar_core::httpd::TlsOpt::Files(c, k) }
                    else { mimar_core::httpd::TlsOpt::Off }
                };
                mimar_core::httpd::serve_http_full(&path, &addr, replica, &token, &read_token, rate, tls)?;
            }
            "serve" => {
                let model = opt(&args, "--model", "llama3.2:3b");
                let engine = pick_engine(&args, &model);
                mimar_core::mcp::serve(&path, engine)?;
            }
            "stats" => {
                let b = Brain::open(Some(&path))?;
                println!("rol     : {}", b.role());
                println!("feature : {}", b.feature());
                println!("niyet   : {}", b.cache_len());
                println!("guard   : {}", b.guard_patterns().len());
            }
            _ => return Err(format!("bilinmeyen komut: {}", cmd)),
        }
        Ok(())
    })();

    if let Err(e) = result {
        eprintln!("hata: {}", e);
        process::exit(1);
    }
}
