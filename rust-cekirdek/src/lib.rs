//! Mimar gömülü çekirdek (libmimar) — Rust, sıfır bağımlılık.
//! ANA MOTOR burada: format oku/yaz, embedder, seçici, store, model motoru
//! (Ollama/yerel), ML, indeks, MCP. Python ve diğer diller opsiyonel ince binding.
//! GUI AYRI bir program (src/bin/gui.rs, eframe) — çekirdeği kirletmez.

pub mod hash;
pub mod json;
pub mod embed;
pub mod store;
pub mod ml;
pub mod index;
pub mod hnsw;
pub mod ffi;
pub mod embedder;
pub mod modality;
pub mod quant;
pub mod crypto;
pub mod safety;

// --- Ağ/thread gerektiren modüller (gömülü/wasm çekirdekte YOK) ---
#[cfg(feature = "net")]
pub mod engine;
#[cfg(feature = "net")]
pub mod mcp;
#[cfg(feature = "net")]
pub mod httpd;
#[cfg(feature = "net")]
pub mod remote;
#[cfg(feature = "net")]
pub mod distill;
