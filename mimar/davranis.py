"""
Mimar — Davranış Damıtma (taşınabilir, modelden bağımsız davranış).

Fikir: birileri yüzlerce sohbetle bir modele istediğini yaptırır (ya da LoRA
verisi hazırlar). Mimar bu sohbetleri hem EĞİTİM hem TEST olarak değerlendirir,
tek bir `.mimar` dosyasına damıtır; o dosya başka bir modele takıldığında model
"aynı şeye" dönüşür — ve bu "aynılık" eval'le ÖLÇÜLÜR, iddia edilmez.

Sadakat (fidelity) bir SPEKTRUMDUR, sihir değil:
  - Yakalanan niyetler: dosyadan birebir yeniden üretilir (bağlam katmanı).
  - Parafrazlar: anlamsal eşleşmeyle yeniden üretilir (genelleme).
  - En yüksek sadakat: aynı model + LoRA; farklı model için LoRA yeniden-damıtılır
    (bu katman GERÇEK eğitilebilir model ister — burada iskelet olarak işaretlenir).

Bu modül saf mekanizmayı ve ÖLÇÜM iskeletini sağlar; gerçek LLM sadakati gerçek
modelle ölçülür.
"""

from __future__ import annotations

from typing import Callable, Optional

from .engines.base import Message
from .selector import cosine


def capture(engine, prompts: list[str], system: str = "") -> list[tuple]:
    """Bir modelin davranışını yakala: her istem için modelin çıktısını kaydet.
    'Yüzlerce sohbet' bu çiftlerdir — eğitim verisi + regresyon testi."""
    pairs = []
    for p in prompts:
        msgs = []
        if system:
            msgs.append(Message("system", system))
        msgs.append(Message("user", p))
        out = engine.complete(msgs).text
        pairs.append((p, out))
    return pairs


def ingest(brain, pairs: list[tuple], persona: Optional[str] = None) -> dict:
    """Yakalanan sohbetleri beyne damıt: her çift güvenilir niyet + eval olur.
    persona verilirse role'e eklenir (stil ipucu)."""
    if persona:
        brain.data["role"] = (brain.data.get("role", "") + "\n" + persona).strip()
    for prompt, answer in pairs:
        brain.teach(prompt, answer)
    brain._maybe_save()
    return {"ogrenilen": len(pairs),
            "eval": len(brain.data["evals"]),
            "niyet": len(brain.data["cache"])}


def _match(produced: str, expected: str, embedder, thresh: float = 0.9):
    exact = produced.strip() == expected.strip()
    sim = cosine(embedder.embed(produced), embedder.embed(expected))
    return (exact or sim >= thresh), round(sim, 3)


def fidelity(brain, engine, cases: list[tuple], thresh: float = 0.9) -> dict:
    """Beyni verilen motora takıp davranış eval'lerini koştur; "aynılık" oranını
    ÖLÇ. cases: (girdi, beklenen_çıktı) çiftleri."""
    saved_engines, saved_autosave = brain.engines, brain.autosave
    brain.engines, brain.autosave = [engine], False
    details, hit = [], 0
    for inp, expected in cases:
        out = brain.run(inp).text
        ok, sim = _match(out, expected, brain.embedder, thresh)
        hit += 1 if ok else 0
        details.append({"girdi": inp, "uydu": ok, "benzerlik": sim, "cikti": out})
    brain.engines, brain.autosave = saved_engines, saved_autosave
    return {"oran": round(hit / len(cases), 3) if cases else 0.0,
            "uyan": hit, "toplam": len(cases), "detay": details}


def raw_fidelity(engine, system: str, cases: list[tuple], embedder,
                 thresh: float = 0.9) -> dict:
    """Beyin OLMADAN aynı motorun sadakati — karşılaştırma tabanı (baseline)."""
    hit = 0
    for inp, expected in cases:
        msgs = [Message("system", system), Message("user", inp)]
        out = engine.complete(msgs).text
        ok, _ = _match(out, expected, embedder, thresh)
        hit += 1 if ok else 0
    return {"oran": round(hit / len(cases), 3) if cases else 0.0,
            "uyan": hit, "toplam": len(cases)}


def redistill_lora(brain, engine) -> str:
    """Farklı model için LoRA YENİDEN-DAMITMA (en yüksek sadakat katmanı).
    DÜRÜST SINIR: bu, eğitilebilir bir model + eğitim döngüsü gerektirir; burada
    yalnızca format kancası vardır (beyin damıtma verisini taşır, çıktı adapter
    olarak saklanır). Gerçek eğitim entegrasyonu yol haritasında."""
    data_pairs = [(e["prompt"], e["expect"]) for e in brain.data["evals"]]
    # Placeholder: damıtma verisini taşıdığımızı ve adapter yuvasını işaretliyoruz.
    manifest = f"lora-redistill:{getattr(engine, 'name', '?')}:{len(data_pairs)} çift"
    brain.put_adapter(f"redistill-{getattr(engine, 'name', 'x')}",
                      manifest.encode("utf-8"), )
    return ("iskelet: " + manifest +
            "  (gerçek ağırlık eğitimi eğitilebilir model ister)")
