"""
Mimar — Konsolidasyon (şişmeyen beyin).

Bir veritabanı her sorguyu saklamaz; veriyi + istatistiği tutar. Mimar da deneyimi
BİRİKTİRMEMELİ, SIKIŞTIRMALI — yoksa milyon konuşmada dosya şişer (dandik not
defteri). İnsan beynindeki konsolidasyon (uyku) gibi: ham epizotlar geçicidir,
kalan şey damıtılmış öz.

Üç mekanizma:
  1) Birleştirme : anlamca çok yakın niyetler (cosine ≥ eşik) tek prototipte
     toplanır (tekrar/isabet toplanır, vektör ağırlıklı ortalama).
  2) Budama     : kapasite aşılırsa uzun-kuyruktaki DÜŞÜK FAYDALI geçici niyetler
     atılır. Öğretilmiş/öğrenilmiş/engelli niyetler KORUNUR (davranış ve güvenlik
     bozulmaz → doğruluk korunur).
  3) Sınırlama  : epizodik hafıza ve günlük üst sınıra çekilir.

Böylece dosya trafikle DOĞRUSAL büyümez; davranış ve güvenlik aynen kalır.
"""

from __future__ import annotations

import math
from typing import Optional

from .selector import cosine


def _wmean(pairs: list) -> list:
    """Ağırlıklı (n ile) ortalama vektör, L2-normalize."""
    pairs = [(v, n) for v, n in pairs if v]
    if not pairs:
        return []
    dim = len(pairs[0][0])
    acc = [0.0] * dim
    for v, n in pairs:
        for i, x in enumerate(v):
            acc[i] += x * n
    norm = math.sqrt(sum(x * x for x in acc))
    return [x / norm for x in acc] if norm > 0 else acc


def _protected(c: dict) -> bool:
    """Korunan niyet: öğretilmiş, öğrenilmiş ya da engelli (güvenlik)."""
    return bool(c.get("learned") or c.get("source") == "taught" or c.get("blocked"))


def consolidate(brain, max_intents: Optional[int] = None, max_memory: int = 50,
                max_log: int = 50, merge_sim: Optional[float] = None) -> dict:
    from .brain import SAME_SIM
    merge_sim = SAME_SIM if merge_sim is None else merge_sim
    cache = brain.data["cache"]
    before = len(cache)
    before_mem = len(brain.data["memory"])

    # Engelli niyetler birleştirmeye GİRMEZ (blok korunur)
    blocked = {k: c for k, c in cache.items() if c.get("blocked")}
    items = [(k, c) for k, c in cache.items() if not c.get("blocked")]

    # 1) Açgözlü kümeleme (anlamsal yakınlık)
    used = set()
    clusters = []
    for i, (k, c) in enumerate(items):
        if k in used:
            continue
        cluster = [(k, c)]
        used.add(k)
        for k2, c2 in items[i + 1:]:
            if k2 in used:
                continue
            if cosine(c.get("vec", []), c2.get("vec", [])) >= merge_sim:
                cluster.append((k2, c2))
                used.add(k2)
        clusters.append(cluster)

    merged_cache = {}
    for cluster in clusters:
        if len(cluster) == 1:
            k, c = cluster[0]
            merged_cache[k] = c
            continue
        kw, cw = max(cluster, key=lambda kc: (kc[1].get("reward", 0),
                                              kc[1].get("repeats", 0)))
        m = dict(cw)
        m["repeats"] = sum(c.get("repeats", 0) for _, c in cluster)
        m["hits"] = sum(c.get("hits", 0) for _, c in cluster)
        m["reward"] = max(c.get("reward", 0) for _, c in cluster)
        m["n"] = sum(c.get("n", 1) for _, c in cluster)
        m["verified"] = all(c.get("verified", True) for _, c in cluster)
        m["learned"] = any(c.get("learned") for _, c in cluster)
        m["vec"] = _wmean([(c.get("vec"), c.get("n", 1)) for _, c in cluster])
        merged_cache[kw] = m
    merged_cache.update(blocked)
    after_merge = len(merged_cache)

    # 2) Budama (kapasite) — korunanlar dokunulmaz
    evicted = 0
    if max_intents and len(merged_cache) > max_intents:
        protected = {k: c for k, c in merged_cache.items() if _protected(c)}
        others = [(k, c) for k, c in merged_cache.items() if k not in protected]
        # Fayda: ödül > isabet > tekrar > yenilik
        others.sort(key=lambda kc: (kc[1].get("reward", 0), kc[1].get("hits", 0),
                                    kc[1].get("repeats", 0), kc[1].get("ts", 0)),
                    reverse=True)
        keep = dict(protected)
        for k, c in others:
            if len(keep) >= max_intents:
                break
            keep[k] = c
        evicted = len(merged_cache) - len(keep)
        merged_cache = keep

    brain.data["cache"] = merged_cache

    # 3) Sınırlama: hafıza + günlük
    mem = brain.data["memory"]
    mem.sort(key=lambda m: (m.get("weight", 0), m.get("used", 0)), reverse=True)
    brain.data["memory"] = mem[:max_memory]
    brain.data["log"] = brain.data["log"][-max_log:]

    brain._maybe_save()
    return {"niyet_önce": before, "birleşme_sonrası": after_merge,
            "niyet_sonra": len(merged_cache),
            "birleşen": before - after_merge, "budanan": evicted,
            "hafıza_önce": before_mem, "hafıza_sonra": len(brain.data["memory"]),
            "korunan": sum(1 for c in merged_cache.values() if _protected(c))}
