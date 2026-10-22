"""
Mimar — Planlayıcı / Optimizasyoncu (projenin kalbi).

SQLite'ın sorgu planlayıcısı gibi çalışır: niyeti + sorguyu alır, birkaç ADAY
plan üretir, her birinin maliyetini (beynin istatistikleriyle düzeltilmiş) tahmin
eder ve politikayı (risk, para, gecikme) karşılayan EN UCUZ planı seçer.

Aday planlar — ucuzdan pahalıya:
  1) CACHE planı  : bu niyet daha önce derlenmişse, hazır cevabı getir (bedava).
  2) RULE planı   : deterministik bir kural eşleşiyorsa, model çağırma.
  3) MODEL planı  : dayanak topla + uygun güçte motorla sentezle + doğrula + öğren.

Seçim bir MALİYET MODELİNE dayanır; "ucuz" plan ancak güvenilirse (önbellek taze,
kural eşleşiyor) seçilir. Risk 'high' ise ucuz planlar elenir, güçlü motor + zorunlu
doğrulama şart koşulur. Böylece "ne zaman kural yeter, ne zaman model" kararı
elle değil, optimize edilerek verilir.
"""

from __future__ import annotations

from typing import Optional

from .security import Policy
from .vm import (CACHE_GET, COMMIT, MODEL, RECALL, RETRIEVE, RULE, TOOL, VERIFY,
                 Cost, Op, OP_BASE, Plan)


class Planner:
    def __init__(self, brain, engines: list, policy: Policy):
        self.brain = brain
        self.engines = engines
        self.policy = policy

    # ---- aday plan üretimi ----------------------------------------------
    def candidates(self, query: str) -> list[Plan]:
        plans: list[Plan] = []

        # 1) CACHE planı — niyet daha önce derlenmiş mi?
        if self.brain.cache_get(query):
            ops = [Op(CACHE_GET, "bu niyet daha önce derlenmiş, hazır cevap"),
                   Op(VERIFY, "derlenmiş cevap yine de guard'lardan geçer")]
            plans.append(self._cost(query, ops,
                                    note="derlenmiş yol (kullanıldıkça ucuzladı)"))

        # 2) RULE planı — geliştiricinin yazdığı reflex kuralı eşleşiyor mu?
        if self.brain.match_reflex(query):
            ops = [Op(RULE, "deterministik reflex kuralı doğrudan cevaplıyor"),
                   Op(VERIFY, "kural çıktısı guard'lardan geçer")]
            plans.append(self._cost(query, ops, note="reflex kural yolu (model yok)"))

        # 3) MODEL planı — tam biliş hattı
        ops = [Op(RECALL, "ilgili hafıza/istatistik getirilir")]
        for t in self.brain.triggered_tools(query):
            ops.append(Op(TOOL, f"'{t}' deterministik veri sağlıyor",
                          {"tool": t}))
        if self.brain.has_knowledge():
            ops.append(Op(RETRIEVE, "bilgi indeksinden dayanak", {"k": 3}))
        engine, tier, strength = self._choose_engine()
        ops.append(Op(MODEL, f"'{tier}' motoruyla sentez (güç {strength:.2f})",
                      {"engine": tier}))
        ops.append(Op(VERIFY, "üretilen yanıt guard'lara göre doğrulanır"))
        ops.append(Op(COMMIT, "cevap önbelleğe/kurala derlenir (uzmanlaşma)"))
        plans.append(self._cost(query, ops, note=f"tam hat ({tier})"))

        return plans

    # ---- seçim -----------------------------------------------------------
    def plan(self, query: str) -> Plan:
        cands = self.candidates(query)
        pool = [p for p in cands if self._feasible(p)] or cands

        def has_model(p):
            return any(o.name == MODEL for o in p.ops)

        if self.policy.risk == "high":
            # Yüksek risk: kestirmeyi eleme, güçlü model + doğrulama ZORUNLU.
            strict = [p for p in pool if has_model(p)]
            return min(strict or pool, key=lambda p: (p.est.usd, p.est.ms))
        # Risk normal/düşük: amaç SÖZLÜK-SIRALI — önce modeli ATLA (derlenmiş
        # cevap ücretsiz + deterministik + modele bağımsızdır), sonra para, sonra
        # gecikme. Böylece hızlı/ucuz bir mock motor bile derlenmiş yolu bastıramaz;
        # "kural/önbellek yetiyorsa modeli çağırma" ilkesi korunur.
        compiled = [p for p in pool if not has_model(p)]
        chosen_pool = compiled or pool
        return min(chosen_pool, key=lambda p: (p.est.usd, p.est.ms))

    # ---- maliyet modeli --------------------------------------------------
    def _cost(self, query: str, ops: list[Op], note: str = "") -> Plan:
        total = Cost()
        for op in ops:
            base = OP_BASE.get(op.name, Cost())
            # Beynin ölçtüğü gerçek gecikmeyle düzelt (ANALYZE etkisi)
            obs = self.brain.op_latency(op.name)
            est = Cost(base.usd, obs if obs is not None else base.ms)
            if op.name == MODEL:
                e = self._engine_by_tier(op.args.get("engine"))
                est = est + Cost(getattr(e, "usd_per_call", 0.0),
                                 getattr(e, "latency_ms", 0.0))
            op.est = est
            total = total + est
        return Plan(query, ops, total, note)

    def _feasible(self, plan: Plan) -> bool:
        if plan.est.usd > self.policy.max_cost_usd:
            return False
        if plan.est.ms > self.policy.max_latency_ms:
            return False
        return True

    def _choose_engine(self):
        e, tier, strength = self._best_engine()
        return e, tier, strength

    def _best_engine(self):
        avail = [e for e in self.engines if _try(e.available)]
        pool = [e for e in avail
                if getattr(e, "strength", 0.5) >= self.policy.min_strength] or avail
        if not pool:
            return None, "yok", 0.0
        if self.policy.risk == "high":
            e = max(pool, key=lambda x: getattr(x, "strength", 0.5))
        else:
            e = min(pool, key=lambda x: getattr(x, "usd_per_call", 0.0))
        return e, getattr(e, "tier", getattr(e, "name", "?")), getattr(e, "strength", 0.5)

    def _engine_by_tier(self, tier):
        for e in self.engines:
            if getattr(e, "tier", getattr(e, "name", "?")) == tier:
                return e
        return self.engines[0] if self.engines else None


def _try(fn) -> bool:
    try:
        return bool(fn())
    except Exception:
        return False
