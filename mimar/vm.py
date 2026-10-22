"""
Mimar — Biliş Sanal Makinesi (Cognition VM).

SQLite, SQL'i VDBE adlı bir sanal makinede koşan bayt koduna derler. Mimar da
niyeti, bir biliş baytkoduna derler: RULE, RECALL, RETRIEVE, TOOL, ASK, MODEL,
VERIFY, REFLECT, CACHE_GET, COMMIT operatörlerinden oluşan bir plan. Bu modül o
operatörleri ve onları çalıştıran VM'i tanımlar.

Her operatörün bir MALİYET tahmini vardır (gecikme, para, güç). Planı optimize
eden `planner`, bu maliyetlere ve beynin biriktirdiği istatistiklere bakarak en
ucuz ama riski karşılayan planı seçer. VM çalışırken GERÇEK maliyeti ölçer ve
istatistikleri günceller — böylece beyin kullanıldıkça tahminleri keskinleşir.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .engines.base import Completion, Message
from .security import Budget, Policy, SecurityError, check_guards, redact

# Operatör adları (biliş baytkodu komut seti)
RULE = "RULE"            # deterministik kural doğrudan cevaplar
RECALL = "RECALL"        # hafıza/istatistik getir
RETRIEVE = "RETRIEVE"    # bilgi indeksinden dayanak getir (RAG)
TOOL = "TOOL"            # bağlı fonksiyon çağır
ASK = "ASK"              # eksik bilgi için insana sor
MODEL = "MODEL"          # seçilen motordan sentez
VERIFY = "VERIFY"        # çıktıyı guard'lara/tutarlılığa göre denetle
REFLECT = "REFLECT"      # başarısızlıkta planı gözden geçir (sınırlı)
CACHE_GET = "CACHE_GET"  # derlenmiş (önceden öğrenilmiş) cevabı getir
COMMIT = "COMMIT"        # öğrenileni beyne yaz (önbellek/kural/istatistik)


@dataclass
class Cost:
    """Bir operatörün veya planın maliyeti."""
    usd: float = 0.0
    ms: float = 0.0

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(self.usd + other.usd, self.ms + other.ms)

    def as_dict(self) -> dict:
        return {"usd": round(self.usd, 6), "ms": round(self.ms, 2)}


# Operatörlerin taban maliyet tahmini (motor hariç; motor maliyeti ayrı eklenir)
OP_BASE: dict[str, Cost] = {
    RULE:      Cost(0.0, 0.05),
    CACHE_GET: Cost(0.0, 0.1),
    RECALL:    Cost(0.0, 0.3),
    RETRIEVE:  Cost(0.0, 1.5),
    TOOL:      Cost(0.0, 5.0),
    ASK:       Cost(0.0, 0.0),   # gecikme insana bağlı, modellenmez
    VERIFY:    Cost(0.0, 0.5),
    REFLECT:   Cost(0.0, 0.2),
    COMMIT:    Cost(0.0, 0.4),
    MODEL:     Cost(0.0, 0.0),   # gerçek maliyet seçilen motordan gelir
}


@dataclass
class Op:
    """Plandaki tek bir biliş operatörü."""
    name: str
    why: str = ""
    args: dict = field(default_factory=dict)
    est: Cost = field(default_factory=Cost)

    def __str__(self) -> str:
        a = f" {self.args}" if self.args else ""
        c = f"  ~{self.est.ms:.0f}ms"
        if self.est.usd:
            c += f"/${self.est.usd:.4f}"
        return f"{self.name:<9}{a}  — {self.why}{c}"


@dataclass
class Plan:
    """Derlenmiş biliş planı — EXPLAIN ile incelenebilir."""
    query: str
    ops: list[Op]
    est: Cost = field(default_factory=Cost)
    note: str = ""

    def __str__(self) -> str:
        head = f"PLAN «{self.query}»   tahmini ~{self.est.ms:.0f}ms"
        if self.est.usd:
            head += f" / ${self.est.usd:.4f}"
        if self.note:
            head += f"\n     ↳ {self.note}"
        body = "\n".join(f"  {i+1}. {op}" for i, op in enumerate(self.ops))
        return f"{head}\n{body}"

    def as_list(self) -> list[dict]:
        return [{"op": o.name, "why": o.why, "args": o.args,
                 "est": o.est.as_dict()} for o in self.ops]


@dataclass
class Answer:
    text: str
    plan: Plan
    trace: list[dict]
    cost: Cost
    source: str = "model"      # model | cache | rule
    blocked: bool = False

    def __str__(self) -> str:
        return self.text


class VM:
    """Biliş baytkodunu çalıştırır, gerçek maliyeti ölçer, istatistiği günceller."""

    def __init__(self, brain, engines: list, bound_tools: dict[str, Callable],
                 policy: Policy, ask_fn: Optional[Callable[[str], str]] = None):
        self.brain = brain
        self.engines = engines
        self.bound_tools = bound_tools
        self.policy = policy
        self.ask_fn = ask_fn

    def execute(self, plan: Plan) -> Answer:
        budget = Budget(self.policy)
        trace: list[dict] = []
        cost = Cost()
        ctx: list[Message] = []
        answer = ""
        source = "model"
        query = plan.query

        for op in plan.ops:
            t0 = time.perf_counter()

            if op.name == CACHE_GET:
                hit = self.brain.cache_get(query)
                if hit:
                    answer = hit["answer"]
                    source = "rule" if hit.get("learned") else "cache"
                    self.brain.cache_touch(query)
                    trace.append({"op": CACHE_GET, "hit": True,
                                  "sim": hit.get("_sim"), "source": source})
                else:
                    trace.append({"op": CACHE_GET, "hit": False})

            elif op.name == RULE:
                r = self.brain.match_reflex(query)
                if r and r.get("response"):
                    answer = r["response"]
                    source = "rule"
                    trace.append({"op": RULE, "rule_id": r.get("id"),
                                  "kind": r.get("kind")})
                else:
                    trace.append({"op": RULE, "hit": False})

            elif op.name == RECALL:
                mem = self.brain.top_memory(3)
                for m in mem:
                    ctx.append(Message("tool", f"hafıza: {m['text'][:200]}"))
                trace.append({"op": RECALL, "n": len(mem)})

            elif op.name == RETRIEVE:
                hits = self.brain.search_knowledge(query, k=op.args.get("k", 3))
                for h in hits:
                    ctx.append(Message("tool", f"dayanak: {h}"))
                trace.append({"op": RETRIEVE, "hits": hits})

            elif op.name == TOOL:
                name = op.args["tool"]
                if not self.policy.tool_allowed(name, set(self.bound_tools)):
                    trace.append({"op": TOOL, "tool": name, "skip": "izin yok"})
                else:
                    budget.spend_tool()
                    try:
                        res = str(self.bound_tools[name](query))
                    except Exception as e:
                        res = f"(araç hatası: {e})"
                    ctx.append(Message("tool", f"{name} -> {res}"))
                    trace.append({"op": TOOL, "tool": name, "result": res})

            elif op.name == ASK:
                if self.ask_fn:
                    ans = self.ask_fn(op.args.get("prompt", "Açar mısınız?"))
                    ctx.append(Message("user", ans))
                    trace.append({"op": ASK, "answer": ans})

            elif op.name == MODEL:
                budget.spend_model()
                engine = self._pick_engine(op.args.get("engine"))
                messages = self._build_messages(query, ctx)
                comp: Completion = engine.complete(messages)
                answer = comp.text
                source = "model"
                mc = _engine_cost(engine, comp)
                cost = cost + mc
                trace.append({"op": MODEL, "engine": comp.engine,
                              "tokens": comp.tokens, "cost": mc.as_dict()})

            elif op.name == VERIFY:
                forbid = self.brain.guard_patterns()
                hit = check_guards(answer, forbid)
                if hit:
                    trace.append({"op": VERIFY, "blocked": hit})
                    self._record(plan, trace, blocked=True)
                    return Answer("(Yanıt bir güvenlik kuralını ihlal ettiği için "
                                  "engellendi.)", plan, trace, cost,
                                  source=source, blocked=True)
                trace.append({"op": VERIFY, "ok": True})

            elif op.name == COMMIT:
                if self.policy.dev_mode or source == "model":
                    promoted = self.brain.learn(query, answer, source=source)
                    trace.append({"op": COMMIT, "learned": promoted})

            dt = (time.perf_counter() - t0) * 1000.0
            cost = cost + Cost(0.0, dt)
            self.brain.observe_op(op.name, dt)

        self._record(plan, trace, blocked=False)
        return Answer(answer, plan, trace, cost, source=source)

    # ---- yardımcılar -----------------------------------------------------
    def _pick_engine(self, want_tier: Optional[str]):
        avail = [e for e in self.engines if _try(e.available)]
        if not avail:
            raise SecurityError("Kullanılabilir motor yok.")
        if want_tier:
            for e in avail:
                if getattr(e, "tier", "?") == want_tier:
                    return e
        # Politikayı karşılayan en ucuz motor
        feasible = [e for e in avail
                    if getattr(e, "strength", 0.5) >= self.policy.min_strength]
        pool = feasible or avail
        return min(pool, key=lambda e: getattr(e, "usd_per_call", 0.0))

    def _build_messages(self, query: str, ctx: list[Message]) -> list[Message]:
        system = self.brain.system_prompt()
        msgs = [Message("system", system)]
        msgs.extend(ctx)
        q = redact(query) if self.policy.redact_pii else query
        msgs.append(Message("user", q))
        return msgs

    def _record(self, plan: Plan, trace: list[dict], blocked: bool) -> None:
        self.brain.log_run(plan.query, plan.as_list(), trace)


def _engine_cost(engine, comp: Completion) -> Cost:
    usd = getattr(engine, "usd_per_call", 0.0)
    ms = getattr(engine, "latency_ms", 0.0)
    return Cost(usd, ms)


def _try(fn) -> bool:
    try:
        return bool(fn())
    except Exception:
        return False
