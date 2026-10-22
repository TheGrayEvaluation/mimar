//! Mimar C ABI (FFI) — her dilden çağrılabilir kütüphane yüzeyi.
//!
//! `cdylib` olarak derlenir (mimar_core.dll / libmimar_core.so / .dylib). C, Python
//! (ctypes), Node (ffi-napi), Go (cgo), C# (P/Invoke) buradan çağırır. Felsefe:
//! **Mimar YÖNLENDİRİR, host ÇALIŞTIRIR** — bu yüzden FFI model çağırmaz; resolve/
//! recall/teach sunar, model/araç çağrısını host yapar.
//!
//! Dize döndüren fonksiyonların sonucu `mimar_free_string` ile serbest bırakılmalı.

use crate::json::{self, J};
use crate::store::Brain;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

unsafe fn rstr<'a>(p: *const c_char) -> &'a str {
    if p.is_null() { return ""; }
    CStr::from_ptr(p).to_str().unwrap_or("")
}

fn out(s: String) -> *mut c_char {
    CString::new(s).unwrap_or_else(|_| CString::new("").unwrap()).into_raw()
}

/// Beyni aç (yoksa oluştur). Hata → NULL. `mimar_close` ile kapat.
#[no_mangle]
pub extern "C" fn mimar_open(path: *const c_char) -> *mut Brain {
    let p = unsafe { rstr(path) };
    let p = if p.is_empty() { None } else { Some(p) };
    match Brain::open(p) {
        Ok(b) => Box::into_raw(Box::new(b)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn mimar_close(b: *mut Brain) {
    if !b.is_null() { unsafe { drop(Box::from_raw(b)); } }
}

/// Tipli direktif (JSON): Fact/Rule/Tool/Stale/Miss. MODEL ÇAĞIRMAZ.
/// Host, JSON'a bakıp ne yapacağına karar verir (bedava cevap mı, araç mı, model mi).
#[no_mangle]
pub extern "C" fn mimar_resolve(b: *mut Brain, q: *const c_char) -> *mut c_char {
    if b.is_null() { return std::ptr::null_mut(); }
    let brain = unsafe { &*b };
    let d = brain.resolve(unsafe { rstr(q) });
    out(json::write(&d.to_json()))
}

/// Basit: biliyorsa cevabı (string) döndür, bilmiyorsa NULL. (resolve'un kısa yolu.)
#[no_mangle]
pub extern "C" fn mimar_recall(b: *mut Brain, q: *const c_char) -> *mut c_char {
    if b.is_null() { return std::ptr::null_mut(); }
    let brain = unsafe { &*b };
    match brain.recall(unsafe { rstr(q) }) {
        Some(h) => out(h.answer),
        None => std::ptr::null_mut(),
    }
}

/// Öğret (girdi→çıktı, yetkili/kalıcı) + diske kaydet. 0=ok, <0=hata.
#[no_mangle]
pub extern "C" fn mimar_teach(b: *mut Brain, prompt: *const c_char, answer: *const c_char) -> i32 {
    if b.is_null() { return -1; }
    let brain = unsafe { &mut *b };
    brain.teach(unsafe { rstr(prompt) }, unsafe { rstr(answer) });
    if brain.save().is_ok() { 0 } else { -2 }
}

/// Geri bildirim: good=1 pozitif, 0 → niyeti engelle. 0=ok.
#[no_mangle]
pub extern "C" fn mimar_feedback(b: *mut Brain, q: *const c_char, good: i32) -> i32 {
    if b.is_null() { return -1; }
    let brain = unsafe { &mut *b };
    brain.feedback(unsafe { rstr(q) }, good != 0);
    if brain.save().is_ok() { 0 } else { -2 }
}

/// İstatistik (JSON: role/intents/guards/feature).
#[no_mangle]
pub extern "C" fn mimar_stats(b: *mut Brain) -> *mut c_char {
    if b.is_null() { return std::ptr::null_mut(); }
    let brain = unsafe { &*b };
    let j = J::Obj(vec![
        ("role".into(), J::s(&brain.role())),
        ("intents".into(), J::n(brain.cache_len() as f64)),
        ("guards".into(), J::n(brain.guard_patterns().len() as f64)),
        ("feature".into(), J::s(&brain.feature())),
    ]);
    out(json::write(&j))
}

/// Dize döndüren fonksiyonların sonucunu serbest bırak.
#[no_mangle]
pub extern "C" fn mimar_free_string(s: *mut c_char) {
    if !s.is_null() { unsafe { drop(CString::from_raw(s)); } }
}

// ---- Çok-kiplilik (KİP-AGNOSTİK): host medyayı parmaklar → vektör → teach/resolve ----

use std::os::raw::{c_double, c_int};
use std::slice;

/// Gri piksel düzlemini (w×h, 0..=255) algısal vektöre parmakla. Çıktı `out`'a yazılır
/// (uzunluk = dönen değer; 256). `out` en az 256 `double` tutmalı. Hata → 0.
#[no_mangle]
pub extern "C" fn mimar_embed_image(gray: *const u8, w: usize, h: usize,
                                    out_vec: *mut c_double, out_cap: usize) -> c_int {
    if gray.is_null() || out_vec.is_null() || w == 0 || h == 0 { return 0; }
    let pix = unsafe { slice::from_raw_parts(gray, w * h) };
    let v = crate::modality::embed_image(pix, w, h);
    if out_cap < v.len() { return 0; }
    unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), out_vec, v.len()); }
    v.len() as c_int
}

/// PCM örneklerini (mono f64, -1..=1) algısal ses vektörüne parmakla. Çıktı `out`'a.
#[no_mangle]
pub extern "C" fn mimar_embed_audio(samples: *const c_double, n: usize,
                                    out_vec: *mut c_double, out_cap: usize) -> c_int {
    if samples.is_null() || out_vec.is_null() || n == 0 { return 0; }
    let s = unsafe { slice::from_raw_parts(samples, n) };
    let v = crate::modality::embed_audio(s);
    if out_cap < v.len() { return 0; }
    unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), out_vec, v.len()); }
    v.len() as c_int
}

/// Vektör-tabanlı öğret (medya/veri): anahtar + vektör + cevap. Diske kaydeder. 0=ok.
#[no_mangle]
pub extern "C" fn mimar_teach_vec(b: *mut Brain, key: *const c_char,
                                  vec: *const c_double, n: usize, answer: *const c_char) -> i32 {
    if b.is_null() || vec.is_null() { return -1; }
    let brain = unsafe { &mut *b };
    let qv = unsafe { slice::from_raw_parts(vec, n) }.to_vec();
    brain.teach_vec(unsafe { rstr(key) }, &qv, unsafe { rstr(answer) });
    if brain.save().is_ok() { 0 } else { -2 }
}

/// Vektör-tabanlı resolve (medya/veri): tipli direktif JSON. MODEL ÇAĞIRMAZ.
#[no_mangle]
pub extern "C" fn mimar_resolve_vec(b: *mut Brain, vec: *const c_double, n: usize) -> *mut c_char {
    if b.is_null() || vec.is_null() { return std::ptr::null_mut(); }
    let brain = unsafe { &*b };
    let qv = unsafe { slice::from_raw_parts(vec, n) }.to_vec();
    out(json::write(&brain.resolve_vec(&qv, None).to_json()))
}

/// Güvenlik taraması: prompt-injection/jailbreak risk etiketleri (JSON dizi). Boş dizi = temiz.
#[no_mangle]
pub extern "C" fn mimar_scan(text: *const c_char) -> *mut c_char {
    let tags = crate::safety::scan_injection(unsafe { rstr(text) });
    out(json::write(&J::Arr(tags.iter().map(|t| J::s(t)).collect())))
}

#[no_mangle]
pub extern "C" fn mimar_version() -> *mut c_char { out("0.1.0".into()) }
