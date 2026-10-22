"""
Mimar — Beyin Birleştirme (bağımsız ATTACH + derlenmiş MERGE).

Her .mimar beyni bağımsızdır; ama birleştirilerek YENİ bir beyin de üretilebilir.
SQLite'ın iki mekanizmasının karşılığı:

  - ATTACH : birden çok beyni çalışma anında yan yana kullan; her dosya kendi
             başına kalır (bağımsızlık korunur). Bellekte birleşik görünüm.
  - MERGE  : kuralların/araçların/bilginin/derlenmiş niyetlerin/istatistiğin
             birleşimi olan YENİ bir .mimar dosyası derle — deterministik eser.

Güven: her kaynak, birleşmeye imzalı bir PROVENANS kaydıyla girer (kaynak adı +
içerik hash'i + zaman). Her birleşen öğe kaynağıyla etiketlenir (`_src`). Çakışmalar
politikayla çözülür ve raporlanır — sessizce ezme yok.

Güvenlik notu: araç KODU asla taşınmaz, yalnızca bildirim. Bu yüzden birleştirmek
host'un vermediği bir yeteneği ele geçiremez. (Kriptografik imzalama — anahtarlı —
sertleştirme adımı olarak yol haritasında; burada içerik-hash provenansı var.)
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Optional

from .brain import Brain
from .engines import Echo

_CORE = ("role", "rules", "tools", "knowledge", "cache", "evals")


def content_hash(brain: Brain) -> str:
    core = {k: brain.data.get(k) for k in _CORE}
    blob = json.dumps(core, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _better(a: dict, b: dict) -> dict:
    """Çakışan derlenmiş niyet: yüksek ödül > tekrar > yenilik kazanır."""
    ka = (a.get("reward", 0), a.get("repeats", 0), a.get("ts", 0))
    kb = (b.get("reward", 0), b.get("repeats", 0), b.get("ts", 0))
    return a if ka >= kb else b


def merge_new(path: Optional[str], sources: list[Brain], engines=None,
              policy=None) -> tuple[Brain, dict]:
    """Kaynak beyinleri YENİ bir beyne derle. (path=None → bellekte = ATTACH.)"""
    dst = Brain(path, engines or [Echo()], policy=policy, autosave=False)
    dst.data["provenance"] = []
    report = {"sources": [], "conflicts": []}

    # --- rol birleşimi ---
    roles = [s.data.get("role", "").strip() for s in sources]
    merged_role = "\n---\n".join(r for r in roles if r)
    if merged_role:
        dst.data["role"] = merged_role

    for i, s in enumerate(sources):
        tag = f"S{i + 1}"
        prov = {"tag": tag, "source": s.path or "bellek", "hash": content_hash(s),
                "ts": time.time(), "rules": len(s.data["rules"]),
                "tools": len(s.data["tools"]), "knowledge": len(s.data["knowledge"]),
                "intents": len(s.data["cache"])}
        dst.data["provenance"].append(prov)
        report["sources"].append(prov)

        # --- kurallar: birleşim; guard'lar toplamsal (daha kısıtlayıcı kazanır) ---
        seen = {(r["kind"], r.get("text"), r.get("trigger"), r.get("response"))
                for r in dst.data["rules"]}
        for r in s.data["rules"]:
            sig = (r["kind"], r.get("text"), r.get("trigger"), r.get("response"))
            if sig in seen:
                continue
            seen.add(sig)
            nr = dict(r); nr["id"] = dst._next_id("rules"); nr["_src"] = tag
            dst.data["rules"].append(nr)

        # --- araçlar: ada göre; çakışma → ilki kalır, raporlanır ---
        names = {t["name"] for t in dst.data["tools"]}
        for t in s.data["tools"]:
            if t["name"] in names:
                report["conflicts"].append(
                    {"tür": "araç", "ad": t["name"], "çözüm": "ilki korundu",
                     "kaynak": tag})
                continue
            names.add(t["name"])
            nt = dict(t); nt["_src"] = tag
            dst.data["tools"].append(nt)

        # --- bilgi: hepsini ekle (metne göre tekilleştir) ---
        known = {k["text"] for k in dst.data["knowledge"]}
        for k in s.data["knowledge"]:
            if k["text"] in known:
                continue
            known.add(k["text"])
            nk = dict(k); nk["id"] = dst._next_id("knowledge"); nk["_src"] = tag
            dst.data["knowledge"].append(nk)

        # --- derlenmiş niyetler: anahtara göre; çakışma → politikayla çöz ---
        for key, c in s.data["cache"].items():
            nc = dict(c); nc["_src"] = tag
            if key not in dst.data["cache"]:
                dst.data["cache"][key] = nc
            else:
                cur = dst.data["cache"][key]
                if cur.get("answer") != nc.get("answer"):
                    win = _better(cur, nc)
                    dst.data["cache"][key] = dict(win)
                    report["conflicts"].append(
                        {"tür": "niyet", "sorgu": c.get("query", key)[:40],
                         "çözüm": f"kazanan _src={win.get('_src','?')}",
                         "kaynak": tag})

        # --- eval'ler: birleşim ---
        dst.data["evals"].extend(s.data["evals"])

        # --- adapter blob'ları: ada göre; çakışma → ilki ---
        for bname, blob in s.blobs.items():
            if bname in dst.blobs:
                report["conflicts"].append(
                    {"tür": "adapter", "ad": bname, "çözüm": "ilki korundu",
                     "kaynak": tag})
                continue
            dst.blobs[bname] = blob

    # --- istatistik: op gecikmelerini kaynaklar arası ortala ---
    op_acc: dict[str, list] = {}
    for s in sources:
        for op, ms in s.data.get("stats", {}).get("op_ms", {}).items():
            op_acc.setdefault(op, []).append(ms)
    dst.data["stats"]["op_ms"] = {op: sum(v) / len(v) for op, v in op_acc.items()}
    dst.data["stats"]["runs"] = sum(s.data.get("stats", {}).get("runs", 0)
                                    for s in sources)

    if path:
        dst.save()
    report["merged"] = {"rules": len(dst.data["rules"]),
                        "tools": len(dst.data["tools"]),
                        "knowledge": len(dst.data["knowledge"]),
                        "intents": len(dst.data["cache"]),
                        "evals": len(dst.data["evals"])}
    dst.merge_report = report
    return dst, report


def attach(sources: list[Brain], engines=None) -> Brain:
    """Bağımsız beyinleri bellekte birleşik görünümle kullan (dosyalar değişmez)."""
    brain, _ = merge_new(None, sources, engines)
    return brain


def verify_provenance(brain: Brain) -> list[dict]:
    """Birleşik beynin kaynak provenans kayıtlarını döndür (kim, hangi hash)."""
    return brain.data.get("provenance", [])
