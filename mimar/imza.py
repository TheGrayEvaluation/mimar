"""
Mimar — Beyin İmzalama (dağıtılabilirlik + güven).

Bir .mimar beyni dağıtıldığında (indirilir, birleştirilir, bir ajana mount edilir)
iki soru kritik: (1) içerik KURCALANDI mı? (2) KİMDEN geldi? İmzalama bunu çözer:
beynin kararlı çekirdeği (rol, kurallar, araç bildirimleri, bilgi, derlenmiş
niyetler, eval'ler, feature) üzerinden bir içerik-hash'i alınır ve imzalanır.
Doğrulama; hash'i yeniden hesaplar (kurcalama tespiti) ve imzayı denetler (kimlik).

Varsayılan algoritma HMAC-SHA256 (bağımlılıksız): anahtarı paylaşan taraflar
bütünlüğü + özgünlüğü doğrular. ASİMETRİK (Ed25519/PKI — anahtar paylaşmadan
herkesin doğrulaması) üretim yükseltmesidir; `cryptography` kuruluysa takılabilir,
burada dürüstçe yalnız HMAC uygulanmıştır.

GÜVENLİK NOTU: imza yalnız BİLDİRİMİ korur; araç KODU zaten dosyada taşınmaz, o
yüzden imzalı bir beyin bile host'un vermediği yeteneği ele geçiremez.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time

# İmzalanan kararlı çekirdek alanları (oynak alanlar hariç: stats/log/memory/
# provenance/signature). feature de dahildir — embedding uzayı kimliğin parçası.
_SIGNED_FIELDS = ("role", "rules", "tools", "knowledge", "cache", "evals", "feature")


def content_hash(brain) -> str:
    core = {k: brain.data.get(k) for k in _SIGNED_FIELDS}
    blob = json.dumps(core, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def sign(brain, key: str, author: str = "") -> dict:
    """Beyni imzala (HMAC-SHA256). İmza dosyaya gömülür."""
    ch = content_hash(brain)
    sig = hmac.new(key.encode("utf-8"), ch.encode("utf-8"),
                   hashlib.sha256).hexdigest()
    brain.data["signature"] = {"alg": "hmac-sha256", "author": author,
                               "content_hash": ch, "sig": sig, "ts": time.time()}
    brain._maybe_save()
    return brain.data["signature"]


def verify(brain, key: str) -> dict:
    """Beyni doğrula: kurcalama + kimlik. {ok, neden, author} döner."""
    s = brain.data.get("signature")
    if not s:
        return {"ok": False, "neden": "imza yok"}
    ch = content_hash(brain)
    if ch != s.get("content_hash"):
        return {"ok": False, "neden": "içerik değişmiş (kurcalama tespit edildi)",
                "author": s.get("author")}
    expect = hmac.new(key.encode("utf-8"), ch.encode("utf-8"),
                      hashlib.sha256).hexdigest()
    ok = hmac.compare_digest(expect, s.get("sig", ""))
    return {"ok": ok, "neden": "geçerli" if ok else "imza/anahtar uyuşmuyor",
            "author": s.get("author")}


def is_signed(brain) -> bool:
    return "signature" in brain.data
