// Mimar — Node binding (C-ABI / cdylib üzerinden).
// Gereksinim: `npm i koffi` + derlenmiş çekirdek (mimar_core.dll/.so/.dylib).
// Felsefe: Mimar YÖNLENDİRİR, host ÇALIŞTIRIR — resolve tipli direktif döndürür,
// modeli/aracı host çağırır.
'use strict';
const path = require('path');

let koffi;
try { koffi = require('koffi'); }
catch { throw new Error("Mimar için 'koffi' gerekli:  npm i koffi"); }

function libPath() {
  if (process.env.MIMAR_LIB) return process.env.MIMAR_LIB;
  const base = path.resolve(__dirname, '..', 'rust-cekirdek', 'target', 'release');
  if (process.platform === 'win32') return path.join(base, 'mimar_core.dll');
  if (process.platform === 'darwin') return path.join(base, 'libmimar_core.dylib');
  return path.join(base, 'libmimar_core.so');
}

const lib = koffi.load(libPath());
const open        = lib.func('void* mimar_open(const char*)');
const close        = lib.func('void mimar_close(void*)');
const resolveRaw  = lib.func('char* mimar_resolve(void*, const char*)');
const recallRaw   = lib.func('char* mimar_recall(void*, const char*)');
const teachRaw    = lib.func('int mimar_teach(void*, const char*, const char*)');
const freeStr     = lib.func('void mimar_free_string(char*)');

function take(ptr) {
  if (!ptr) return null;
  const s = koffi.decode(ptr, 'char*');  // kopyala
  freeStr(ptr);
  return s;
}

class Mimar {
  constructor(brainPath) {
    this.h = open(brainPath);
    if (!this.h) throw new Error('beyin açılamadı: ' + brainPath);
  }
  /** Tipli direktif (JSON): {tip: fact|rule|tool|stale|miss, ...}. Model çağırmaz. */
  resolve(q) { const r = take(resolveRaw(this.h, q)); return r ? JSON.parse(r) : null; }
  /** Biliyorsa cevap (string), bilmiyorsa null. */
  recall(q) { return take(recallRaw(this.h, q)); }
  /** Öğret (yetkili/kalıcı). */
  teach(prompt, answer) { return teachRaw(this.h, prompt, answer) === 0; }
  close() { if (this.h) { close(this.h); this.h = null; } }
}

module.exports = { Mimar };
