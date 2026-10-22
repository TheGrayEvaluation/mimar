//! Mimar GUI — native yönetim konsolu (egui/eframe), tek exe. Tarayıcı yok.
//! DB Browser for SQLite mantığı: .mimar beynini AÇ/YENİ, gör/düzenle/sil,
//! müdahale (engelle), rol & guard & hafıza yönet, sor/öğret. Eğitim değil, YÖNETİM.
//! Motor (mimar_core) sıfır bağımlılık; GUI ayrı binary.

use eframe::egui;
use mimar_core::engine::Engine;
use mimar_core::store::Brain;
use mimar_core::remote::Remote;
use std::time::Instant;

struct App {
    brain: Brain,
    path: String,
    model: String,
    ask_in: String,
    ans: String,
    ans_src: String,
    ans_ms: u128,
    tq: String,
    ta: String,
    role_buf: String,
    guard_in: String,
    mem_in: String,
    sel_query: Option<String>,
    sel_answer: String,
    status: String,
    // ---- Uzak sunucu modu (mimar serve'e REST ile bağlan) ----
    r_url: String,
    r_token: String,
    r_connected: bool,
    r_rows: Vec<(String, String, String, bool, bool)>,
    r_q: String,
    r_a: String,
    r_resolve_in: String,
    r_resolve_out: String,
    r_status: String,
}

impl App {
    fn new(path: &str, model: String) -> Result<Self, String> {
        let brain = Brain::open(Some(path))?;
        let role_buf = brain.role();
        Ok(App {
            brain, path: path.to_string(), model,
            ask_in: String::new(), ans: String::new(), ans_src: String::new(), ans_ms: 0,
            tq: String::new(), ta: String::new(), role_buf, guard_in: String::new(),
            mem_in: String::new(), sel_query: None, sel_answer: String::new(),
            status: String::new(),
            r_url: "http://127.0.0.1:7700".into(), r_token: String::new(),
            r_connected: false, r_rows: Vec::new(), r_q: String::new(), r_a: String::new(),
            r_resolve_in: String::new(), r_resolve_out: String::new(), r_status: String::new(),
        })
    }

    fn remote(&self) -> Remote { Remote::new(&self.r_url, &self.r_token) }
    fn remote_refresh(&mut self) {
        match self.remote().list() {
            Ok(rows) => { self.r_rows = rows; self.r_connected = true;
                          self.r_status = format!("bağlı · {} niyet", self.r_rows.len()); }
            Err(e) => { self.r_connected = false; self.r_status = format!("hata: {}", e); }
        }
    }

    fn open_path(&mut self, p: &str, create: bool) {
        match Brain::open(Some(p)) {
            Ok(mut b) => {
                if create { let _ = b.save(); }
                self.role_buf = b.role();
                self.brain = b;
                self.path = p.to_string();
                self.sel_query = None;
                self.status = format!("açıldı: {}", p);
            }
            Err(e) => self.status = format!("açılamadı: {}", e),
        }
    }

    fn save(&mut self) { let _ = self.brain.save(); }
}

fn tag(src: &str, learned: bool, blocked: bool) -> (egui::Color32, String) {
    if blocked { (egui::Color32::from_rgb(150, 110, 110), "engelli".into()) }
    else if learned { (egui::Color32::from_rgb(255, 180, 84), "öğrenilmiş".into()) }
    else if src == "taught" { (egui::Color32::from_rgb(61, 220, 132), "öğretilmiş".into()) }
    else { (egui::Color32::from_rgb(76, 194, 255), src.to_string()) }
}

const AC: egui::Color32 = egui::Color32::from_rgb(76, 194, 255);

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.heading("🧠 Mimar");
                ui.label(egui::RichText::new("Beyin Yönetim Konsolu").color(egui::Color32::GRAY));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(egui::RichText::new(&self.path).small().color(egui::Color32::GRAY));
                });
            });
            ui.horizontal(|ui| {
                ui.label(format!("{} niyet", self.brain.cache_len()));
                ui.separator();
                ui.label(format!("{} guard", self.brain.guard_patterns().len()));
                ui.separator();
                ui.label(format!("model: {}", self.model));
            });
            ui.add_space(4.0);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {

            // ---- DOSYA ----
            egui::CollapsingHeader::new("📁 Dosya (Yeni / Aç — başka projeye bağlan)")
                .default_open(true).show(ui, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("🆕 Yeni…").clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter("Mimar beyni", &["mimar"])
                            .set_file_name("yeni.mimar").save_file() {
                            self.open_path(&p.to_string_lossy(), true);
                        }
                    }
                    if ui.button("📂 Aç…").clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter("Mimar beyni", &["mimar"]).pick_file() {
                            self.open_path(&p.to_string_lossy(), false);
                        }
                    }
                    ui.label("•");
                    let mut path = self.path.clone();
                    if ui.add(egui::TextEdit::singleline(&mut path).desired_width(360.0)
                        .hint_text("yol yapıştır...")).changed() { self.path = path; }
                    if ui.button("Git").clicked() { let p = self.path.clone(); self.open_path(&p, false); }
                });
                if !self.status.is_empty() {
                    ui.label(egui::RichText::new(&self.status).color(egui::Color32::from_rgb(61, 220, 132)));
                }
            });

            // ---- UZAK SUNUCU (mimar serve'e REST ile bağlan, MySQL Workbench gibi) ----
            egui::CollapsingHeader::new("🌐 Uzak Sunucu (çalışan `mimar serve`'e bağlan)")
                .default_open(false).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("URL:");
                    ui.add(egui::TextEdit::singleline(&mut self.r_url).desired_width(260.0)
                        .hint_text("http://host:7700"));
                    ui.label("Token:");
                    ui.add(egui::TextEdit::singleline(&mut self.r_token).password(true)
                        .desired_width(160.0).hint_text("yazma için (ops.)"));
                    if ui.button("🔌 Bağlan / Yenile").clicked() { self.remote_refresh(); }
                });
                if !self.r_status.is_empty() {
                    let col = if self.r_connected { egui::Color32::from_rgb(61, 220, 132) }
                              else { egui::Color32::from_rgb(230, 120, 120) };
                    ui.label(egui::RichText::new(&self.r_status).color(col).small());
                }
                if self.r_connected {
                    ui.separator();
                    // Uzak sor (resolve — model çağırmaz, sunucu beyninden direktif)
                    ui.horizontal(|ui| {
                        ui.add(egui::TextEdit::singleline(&mut self.r_resolve_in)
                            .hint_text("uzak resolve sorgusu...").desired_width(f32::INFINITY));
                        if ui.button("Resolve").clicked() {
                            let q = self.r_resolve_in.trim().to_string();
                            if !q.is_empty() {
                                self.r_resolve_out = match self.remote().resolve(&q) {
                                    Ok(j) => mimar_core::json::write(&j),
                                    Err(e) => format!("hata: {}", e),
                                };
                            }
                        }
                    });
                    if !self.r_resolve_out.is_empty() {
                        ui.label(egui::RichText::new(&self.r_resolve_out).monospace().small());
                    }
                    // Uzak öğret
                    ui.horizontal(|ui| {
                        ui.add(egui::TextEdit::singleline(&mut self.r_q).hint_text("soru").desired_width(220.0));
                        ui.add(egui::TextEdit::singleline(&mut self.r_a).hint_text("cevap").desired_width(f32::INFINITY));
                        if ui.button("Uzak Öğret").clicked() {
                            let (p, a) = (self.r_q.trim().to_string(), self.r_a.trim().to_string());
                            if !p.is_empty() && !a.is_empty() {
                                match self.remote().teach(&p, &a) {
                                    Ok(_) => { self.r_q.clear(); self.r_a.clear(); self.remote_refresh(); }
                                    Err(e) => self.r_status = format!("öğret hatası: {}", e),
                                }
                            }
                        }
                    });
                    // Uzak niyet listesi + sil
                    egui::Grid::new("remote_intents").striped(true).num_columns(4).min_col_width(90.0).show(ui, |ui| {
                        let rows = self.r_rows.clone();
                        for (q, a, src, learned, blocked) in &rows {
                            ui.label(egui::RichText::new(q).color(AC));
                            ui.label(if a.chars().count() > 60 {
                                format!("{}…", a.chars().take(60).collect::<String>())
                            } else { a.clone() });
                            let (c, t) = tag(src, *learned, *blocked);
                            ui.label(egui::RichText::new(t).color(c).small());
                            if ui.button("🗑").clicked() {
                                let _ = self.remote().forget(q);
                                self.remote_refresh();
                            }
                            ui.end_row();
                        }
                    });
                }
            });

            // ---- SOR ----
            egui::CollapsingHeader::new("💬 Sor").default_open(true).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut self.ask_in)
                        .hint_text("Beyne sor...").desired_width(f32::INFINITY));
                    if ui.button("Sor").clicked() {
                        let q = self.ask_in.trim().to_string();
                        if !q.is_empty() {
                            let eng = Engine::ollama(&self.model);
                            let t0 = Instant::now();
                            match self.brain.ask(&q, &eng, false) {
                                Ok((a, s, _)) => { self.ans = a; self.ans_src = s; self.ans_ms = t0.elapsed().as_millis(); }
                                Err(e) => { self.ans = format!("hata: {}", e); self.ans_src = "hata".into(); }
                            }
                            self.save();
                        }
                    }
                });
                if !self.ans.is_empty() {
                    egui::Frame::group(ui.style()).show(ui, |ui| {
                        ui.label(&self.ans);
                        let (c, _) = tag(&self.ans_src, self.ans_src == "rule", false);
                        ui.label(egui::RichText::new(format!("{} · {}ms", self.ans_src, self.ans_ms)).color(c).small());
                    });
                }
            });

            // ---- NİYETLER (gör/düzenle/sil/müdahale) ----
            let rows = self.brain.intent_rows();
            egui::CollapsingHeader::new(format!("📚 Niyetler ({}) — tıkla, düzenle/sil/engelle", rows.len()))
                .default_open(true).show(ui, |ui| {
                if let Some(q) = self.sel_query.clone() {
                    egui::Frame::group(ui.style()).show(ui, |ui| {
                        ui.label(egui::RichText::new(format!("Seçili: {}", q)).color(AC).strong());
                        ui.add(egui::TextEdit::multiline(&mut self.sel_answer).desired_width(f32::INFINITY).desired_rows(2));
                        ui.horizontal(|ui| {
                            if ui.button("💾 Kaydet").clicked() {
                                self.brain.edit_answer(&q, &self.sel_answer.clone()); self.save();
                                self.status = "cevap güncellendi".into();
                            }
                            if ui.button("🗑 Sil").clicked() {
                                self.brain.delete_intent(&q); self.save();
                                self.sel_query = None; self.status = "silindi".into();
                            }
                            if ui.button("⛔ Engelle").clicked() {
                                self.brain.set_blocked(&q, true); self.save(); self.status = "engellendi".into();
                            }
                            if ui.button("✅ Engeli Kaldır").clicked() {
                                self.brain.set_blocked(&q, false); self.save(); self.status = "engel kaldırıldı".into();
                            }
                            if ui.button("Kapat").clicked() { self.sel_query = None; }
                        });
                    });
                }
                egui::Grid::new("intents").striped(true).num_columns(3).min_col_width(110.0).show(ui, |ui| {
                    for (q, a, src, learned, blocked) in &rows {
                        if ui.add(egui::Label::new(egui::RichText::new(q).color(AC)).sense(egui::Sense::click())).clicked() {
                            self.sel_query = Some(q.clone());
                            self.sel_answer = a.clone();
                        }
                        // UTF-8 güvenli kısaltma: bayt değil, KARAKTER say (çok baytlı 'ü' vb. bölmesin).
                        ui.label(if a.chars().count() > 80 {
                            format!("{}…", a.chars().take(80).collect::<String>())
                        } else { a.clone() });
                        let (c, t) = tag(src, *learned, *blocked);
                        ui.label(egui::RichText::new(t).color(c).small());
                        ui.end_row();
                    }
                });
            });

            // ---- HAFIZA ----
            let mem = self.brain.memory_rows();
            egui::CollapsingHeader::new(format!("🧠 Hafıza ({})", mem.len())).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut self.mem_in).hint_text("yeni hafıza/bilgi...").desired_width(f32::INFINITY));
                    if ui.button("Ekle").clicked() {
                        let t = self.mem_in.trim().to_string();
                        if !t.is_empty() { self.brain.add_memory(&t, "fact"); self.save(); self.mem_in.clear(); }
                    }
                    if ui.button("Temizle").clicked() { self.brain.clear_memory(); self.save(); }
                });
                for m in &mem { ui.label(format!("• {}", m)); }
            });

            // ---- ROL & GUARD & ÖĞRET ----
            egui::CollapsingHeader::new("⚙️ Rol / Prompt, Guard, Öğret").show(ui, |ui| {
                ui.label(egui::RichText::new("Rol / sistem promptu").color(AC));
                ui.add(egui::TextEdit::multiline(&mut self.role_buf).desired_width(f32::INFINITY).desired_rows(2));
                if ui.button("Rolü Kaydet").clicked() {
                    self.brain.set_role(&self.role_buf.clone()); self.save(); self.status = "rol kaydedildi".into();
                }
                ui.separator();
                ui.label(egui::RichText::new("Guard (yasak desenler)").color(AC));
                let guards = self.brain.guard_patterns();
                for g in &guards {
                    ui.horizontal(|ui| {
                        ui.label(format!("⛔ {}", g));
                        if ui.button("sil").clicked() { self.brain.remove_guard(g); self.save(); }
                    });
                }
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut self.guard_in).hint_text("yeni guard deseni").desired_width(300.0));
                    if ui.button("Ekle").clicked() {
                        let g = self.guard_in.trim().to_string();
                        if !g.is_empty() { self.brain.add_guard(&g); self.save(); self.guard_in.clear(); }
                    }
                });
                ui.separator();
                ui.label(egui::RichText::new("Öğret (elle tohumla)").color(AC));
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut self.tq).hint_text("soru").desired_width(240.0));
                    ui.add(egui::TextEdit::singleline(&mut self.ta).hint_text("cevap").desired_width(f32::INFINITY));
                    if ui.button("Öğret").clicked() {
                        let (p, a) = (self.tq.trim().to_string(), self.ta.trim().to_string());
                        if !p.is_empty() && !a.is_empty() { self.brain.teach(&p, &a); self.save(); self.tq.clear(); self.ta.clear(); }
                    }
                });
            });

            });
        });
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "beyin.mimar".to_string());
    let model = args.get(2).cloned().unwrap_or_else(|| "llama3.2:3b".to_string());
    let app = match App::new(&path, model) {
        Ok(a) => a,
        Err(e) => { eprintln!("hata: {}", e); std::process::exit(1); }
    };
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([900.0, 720.0]),
        ..Default::default()
    };
    let _ = eframe::run_native("Mimar — Beyin Yönetim Konsolu", opts,
        Box::new(|_cc| Ok(Box::new(app))));
}
