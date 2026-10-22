"""
Mimar — Python binding (C ABI / cdylib üzerinden, saf ctypes, ek paket yok).

Kurulum: sadece derlenmiş çekirdek gerekir ->  cargo build --release
Kullanım:
    from mimar import Mimar
    with Mimar("beyin.mimar", rol="Destek botu") as m:
        m.teach("iade nasıl", "14 gün içinde")
        print(m.resolve("ürünü geri gönderme"))   # {'tip':'fact','cevap':'14 gün içinde'}

Felsefe: Mimar YÖNLENDİRİR, host ÇALIŞTIRIR. resolve() tipli bir direktif (dict)
döndürür; modeli/aracı çağırmak senin programının işidir.
"""
from __future__ import annotations
import ctypes
import json
import os
import platform

def _default_lib() -> str:
    if "MIMAR_LIB" in os.environ:
        return os.environ["MIMAR_LIB"]
    base = os.path.join(os.path.dirname(__file__), "..", "..", "rust-cekirdek", "target", "release")
    sysname = platform.system()
    if sysname == "Windows":
        return os.path.join(base, "mimar_core.dll")
    if sysname == "Darwin":
        return os.path.join(base, "libmimar_core.dylib")
    return os.path.join(base, "libmimar_core.so")


class Mimar:
    """Tek .mimar beynini saran küçük Python sınıfı."""

    def __init__(self, path: str, rol: str | None = None, lib: str | None = None):
        self._lib = ctypes.CDLL(lib or _default_lib())
        self._decl()
        # rol vermek istenirse CLI olmadan: boş beyni aç, rolü teach ile değil
        # doğrudan açıp bırak (rol yönetimi CLI/GUI'de; burada sade tutuldu).
        self._h = self._lib.mimar_open(path.encode("utf-8"))
        if not self._h:
            raise RuntimeError(f"beyin açılamadı: {path}")

    def _decl(self):
        L = self._lib
        L.mimar_open.restype = ctypes.c_void_p
        L.mimar_open.argtypes = [ctypes.c_char_p]
        L.mimar_close.argtypes = [ctypes.c_void_p]
        L.mimar_resolve.restype = ctypes.c_void_p
        L.mimar_resolve.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.mimar_recall.restype = ctypes.c_void_p
        L.mimar_recall.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.mimar_teach.restype = ctypes.c_int
        L.mimar_teach.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        L.mimar_feedback.restype = ctypes.c_int
        L.mimar_feedback.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        L.mimar_stats.restype = ctypes.c_void_p
        L.mimar_stats.argtypes = [ctypes.c_void_p]
        L.mimar_scan.restype = ctypes.c_void_p
        L.mimar_scan.argtypes = [ctypes.c_char_p]
        L.mimar_version.restype = ctypes.c_void_p
        L.mimar_free_string.argtypes = [ctypes.c_void_p]

    def _take(self, ptr) -> str | None:
        """char* sonucu Python string'e kopyala + Rust tarafında serbest bırak."""
        if not ptr:
            return None
        s = ctypes.cast(ptr, ctypes.c_char_p).value
        self._lib.mimar_free_string(ptr)
        return s.decode("utf-8") if s else None

    # --- genel API ---
    def teach(self, prompt: str, answer: str) -> bool:
        """Yetkili/kalıcı öğret (soru -> cevap)."""
        return self._lib.mimar_teach(self._h, prompt.encode(), answer.encode()) == 0

    def resolve(self, query: str) -> dict:
        """Tipli direktif: {'tip': fact|rule|tool|stale|miss, ...}. Model çağırmaz."""
        return json.loads(self._take(self._lib.mimar_resolve(self._h, query.encode())) or "{}")

    def recall(self, query: str) -> str | None:
        """Biliyorsa cevabı (str), bilmiyorsa None."""
        return self._take(self._lib.mimar_recall(self._h, query.encode()))

    def feedback(self, query: str, good: bool) -> bool:
        return self._lib.mimar_feedback(self._h, query.encode(), 1 if good else 0) == 0

    def stats(self) -> dict:
        return json.loads(self._take(self._lib.mimar_stats(self._h)) or "{}")

    def scan(self, text: str) -> list[str]:
        """Prompt-injection risk etiketleri ([] = temiz)."""
        return json.loads(self._take(self._lib.mimar_scan(text.encode())) or "[]")

    def version(self) -> str:
        return self._take(self._lib.mimar_version()) or "?"

    def close(self):
        if getattr(self, "_h", None):
            self._lib.mimar_close(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
