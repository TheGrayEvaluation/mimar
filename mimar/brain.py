"""
Mimar — Beyin: derlenmiş zihnin taşınabilir eseri (.mimar dosyası).

Bu, projenin "SQLite dosyası" karşılığıdır — ama hazır bir veritabanı değil,
kendi öz formatımız. Bir .mimar dosyası şu bölümlerden oluşan tek bir ikili
kapsayıcıdır:

    MAGIC 'MIMR' | sürüm | bayraklar
    JSON gövde   : rol, kurallar, araç bildirimleri, bilgi, hafıza,
                   ÖNBELLEK (derlenmiş cevaplar), İSTATİSTİK (maliyet modeli)
    BLOB bölümü  : LoRA/adapter tensörleri, gömme vektörleri (ham ikili)

Önemli olan dosyanın kendisi değil, içine DERLENEN şeydir: beyin kullanıldıkça
önbellek ve öğrenilmiş kurallar büyür, istatistikler keskinleşir — yani dosya
zamanla hem daha akıllı hem daha ucuz hale gelir. Zekâyı bir dosya olarak
taşırsınız.

`Brain` nesnesi dosyanın canlı bağlantısıdır (SQLite'ın Connection'ı gibi);
motorları, bağlı araçları ve planlayıcıyı tutar.
"""

from __future__ import annotations

import hashlib
import os
import re
import struct
import time
from typing import Any, Callable, Optional

from .planner import Planner
from .security import Policy, SecurityError
from .selector import Embedder, cosine, mix_prototype
from .vm import VM, Answer, Cost, Plan

MAGIC = b"MIMR"
FORMAT_VERSION = 1
_WORD = re.compile(r"\w+", re.UNICODE)

# Kaç tekrar sonra bir önbellek girdisi kalıcı KURALA derlenir (uzmanlaşma eşiği)
LEARN_THRESHOLD = 2
# İstatistik düzeltmesi için üssel hareketli ortalama katsayısı
EWMA_ALPHA = 0.3
# Anlamsal eşik: bu benzerliğin ÜSTÜNDE derlenmiş cevap güvenle yeniden kullanılır
REUSE_SIM = 0.55
# Bu benzerliğin üstünde "ayrı niyet değil, AYNI niyetin başka ifadesi" sayılır
SAME_SIM = 0.78
# Bu sayıdan çok niyet olunca lineer tarama yerine LSH indeksi devreye girer
INDEX_THRESHOLD = 64


def _tok(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


def _key(query: str) -> str:
    norm = " ".join(_tok(query))
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


def _trigger_hit(trigger: Optional[str], qtok: set[str], query: str) -> bool:
    if not trigger:
        return False
    try:
        if re.search(trigger, query, re.IGNORECASE):
            return True
    except re.error:
        pass
    return bool(set(_tok(trigger)) & qtok)


def _new_data() -> dict:
    return {
        "mimar": FORMAT_VERSION,
        "created": time.time(),
        "role": "Yardımcı bir asistan.",
        "config": {},
        "rules": [],
        "tools": [],
        "knowledge": [],
        "memory": [],
        "evals": [],
        "cache": {},
        "stats": {"op_ms": {}, "runs": 0},
        "log": [],
    }


class Brain:
    """Bir .mimar beynine canlı bağlantı."""

    def __init__(self, path: Optional[str], engines: list,
                 policy: Optional[Policy] = None, autosave: bool = True,
                 embedder=None):
        self.path = path
        self.engines = engines if isinstance(engines, list) else [engines]
        self.policy = policy or Policy()
        self.autosave = autosave
        self.bound_tools: dict[str, Callable] = {}
        self.ask_fn: Optional[Callable[[str], str]] = None
        self.blobs: dict[str, bytes] = {}
        # Embedder VARSAYILAN olarak hash (hiçbir şey değişmez). Güçlü/nöral
        # embedder isteğe bağlı 2. katmandır (opt-in).
        self.embedder = embedder or Embedder()
        if path and os.path.exists(path):
            self._load()
            stored = self.data.get("feature")
            if stored and stored != self.embedder.id:
                # Uzay tutarlılığı: yanlış embedder'la açmak vektörleri geçersiz
                # kılar → SESSİZCE BOZMA, net hata ver. (Dönüştürmek için dosyayı
                # kendi embedder'ıyla aç, sonra set_embedder() çağır.)
                raise ValueError(
                    f"Embedder uyumsuz: dosya '{stored}' ≠ verilen "
                    f"'{self.embedder.id}'. Aynı embedder'la açıp set_embedder() "
                    f"ile dönüştürün.")
        else:
            self.data = _new_data()
        self.data.setdefault("feature", self.embedder.id)
        # Anlamsal indeks (LSH) tembel kurulur; eşik altında lineer tarama kalır.
        self._index = None
        self._indexed_keys: set = set()
        self._index_feature: Optional[str] = None

    # ================= DOSYA FORMATI (öz ikili kapsayıcı) =================
    def _load(self) -> None:
        import json
        with open(self.path, "rb") as f:
            raw = f.read()
        if raw[:4] != MAGIC:
            raise ValueError(f"{self.path} bir Mimar beyni değil (magic yok).")
        off = 4
        ver, flags = struct.unpack_from("<HH", raw, off); off += 4
        (jlen,) = struct.unpack_from("<I", raw, off); off += 4
        self.data = json.loads(raw[off:off + jlen].decode("utf-8")); off += jlen
        (nblob,) = struct.unpack_from("<I", raw, off); off += 4
        self.blobs = {}
        for _ in range(nblob):
            (nl,) = struct.unpack_from("<H", raw, off); off += 2
            name = raw[off:off + nl].decode("utf-8"); off += nl
            (dl,) = struct.unpack_from("<Q", raw, off); off += 8
            self.blobs[name] = raw[off:off + dl]; off += dl

    def save(self, path: Optional[str] = None) -> None:
        import json
        path = path or self.path
        if not path:
            return
        body = json.dumps(self.data, ensure_ascii=False).encode("utf-8")
        out = bytearray()
        out += MAGIC
        out += struct.pack("<HH", FORMAT_VERSION, 0)
        out += struct.pack("<I", len(body))
        out += body
        out += struct.pack("<I", len(self.blobs))
        for name, blob in self.blobs.items():
            nb = name.encode("utf-8")
            out += struct.pack("<H", len(nb)) + nb
            out += struct.pack("<Q", len(blob)) + blob
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            f.write(out)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)   # atomik: SQLite seviyesi dayanıklılık

    def _maybe_save(self) -> None:
        if self.autosave and self.path:
            self.save()

    def close(self) -> None:
        self._maybe_save()

    def __enter__(self): return self
    def __exit__(self, *a): self.close()

    # ================= NİYET BİLDİRİMİ (geliştirici ön yüzü) =============
    def declare(self, role: Optional[str] = None, rules: Optional[list] = None,
                tools: Optional[list] = None, knowledge: Optional[list] = None,
                guards: Optional[list] = None) -> "Brain":
        if role:
            self.data["role"] = role
        for g in guards or []:
            # guard metni hem insan-okur açıklama hem de çıktıda yasak desendir
            self.add_rule(g, kind="guard", forbid=g)
        for r in rules or []:
            if isinstance(r, dict):
                self.add_rule(**r)
            else:
                self.add_rule(r, kind="guide")
        for t in tools or []:
            self.declare_tool(**t) if isinstance(t, dict) else self.declare_tool(t)
        for k in knowledge or []:
            if isinstance(k, dict):
                self.add_knowledge(**k)
            else:
                self.add_knowledge(k)
        self._maybe_save()
        return self

    def tool(self, _fn=None, *, name: Optional[str] = None, trigger: str = "",
             description: str = "", schema: Optional[dict] = None):
        """Hem BİLDİREN hem BAĞLAYAN dekoratör. Bildirim dosyada, kod bellekte."""
        def wrap(fn: Callable):
            tname = name or fn.__name__
            self.declare_tool(tname, description=description or (fn.__doc__ or ""),
                              trigger=trigger, schema=schema)
            self.bound_tools[tname] = fn
            return fn
        return wrap(_fn) if callable(_fn) else wrap

    def bind(self, name: str, fn: Callable) -> None:
        """Bildirilmiş bir aracı bir fonksiyona bağla (kod dosyaya girmez)."""
        self.bound_tools[name] = fn

    def on_ask(self, fn: Callable[[str], str]) -> None:
        self.ask_fn = fn

    # ================= KALICI MUTASYONLAR ================================
    def _next_id(self, coll: str) -> int:
        return (max((x.get("id", 0) for x in self.data[coll]), default=0)) + 1

    def add_rule(self, text: str, kind: str = "guide", trigger: Optional[str] = None,
                 response: Optional[str] = None, forbid: Optional[str] = None,
                 priority: int = 100) -> int:
        rid = self._next_id("rules")
        self.data["rules"].append({"id": rid, "kind": kind, "text": text,
                                   "trigger": trigger, "response": response,
                                   "forbid": forbid, "priority": priority})
        self._maybe_save()
        return rid

    def declare_tool(self, name: str, description: str = "",
                     trigger: Optional[str] = None,
                     schema: Optional[dict] = None) -> None:
        self.data["tools"] = [t for t in self.data["tools"] if t["name"] != name]
        self.data["tools"].append({"name": name, "description": description,
                                    "trigger": trigger, "schema": schema or {}})
        self._maybe_save()

    def add_knowledge(self, text: str, source: str = "",
                      meta: Optional[dict] = None) -> int:
        kid = self._next_id("knowledge")
        self.data["knowledge"].append({"id": kid, "source": source, "text": text,
                                       "meta": meta or {}})
        self._maybe_save()
        return kid

    def remember(self, text: str, kind: str = "episodic") -> int:
        mid = self._next_id("memory")
        self.data["memory"].append({"id": mid, "kind": kind, "text": text,
                                    "weight": 1.0, "created": time.time(),
                                    "used": time.time()})
        return mid

    def put_adapter(self, name: str, blob: bytes) -> None:
        self.blobs[f"adapter/{name}"] = blob
        self._maybe_save()

    def add_eval(self, prompt: str, expect: str = "", name: str = "") -> None:
        self.data["evals"].append({"name": name, "prompt": prompt, "expect": expect})
        self._maybe_save()

    def teach(self, prompt: str, answer: str) -> None:
        """Açık öğretme (davranış yakalama): bir girdi→çıktı çiftini GÜVENİLİR
        kalıcı niyet olarak derle ve aynı anda bir eval (regresyon testi) yap.
        Yakalanan sohbet hem öğretir hem doğrular — davranış damıtmanın temeli."""
        if not prompt or not answer:
            return
        key = _key(prompt)
        exist = self.data["cache"].get(key)
        if exist and exist.get("blocked"):
            return
        self.data["cache"][key] = {
            "query": prompt, "answer": answer,
            "vec": self.embedder.embed(prompt), "n": 1,
            "hits": 0, "repeats": LEARN_THRESHOLD, "verified": True,
            "reward": 1, "blocked": False, "learned": True,
            "source": "taught", "ts": time.time()}
        self.data["evals"].append({"name": "taught", "prompt": prompt,
                                   "expect": answer})
        self._maybe_save()

    # ================= PLANLAYICININ/VM'İN KULLANDIĞI OKUMALAR ===========
    def system_prompt(self) -> str:
        s = self.data.get("role", "")
        guides = [r["text"] for r in self.data["rules"] if r["kind"] == "guide"]
        guards = [r["text"] for r in self.data["rules"] if r["kind"] == "guard"]
        if guides:
            s += "\n\nİlkeler:\n- " + "\n- ".join(guides)
        if guards:
            s += "\n\nKesin yasaklar:\n- " + "\n- ".join(guards)
        mem = self.top_memory(3)
        if mem:
            s += "\n\nGeçmiş bağlam:\n- " + "\n- ".join(m["text"][:200] for m in mem)
        return s

    def match_reflex(self, query: str) -> Optional[dict]:
        """Yalnızca geliştiricinin ELLE yazdığı reflex kuralları (anahtar sözcük/
        regex). Öğrenilmiş niyetler buradan DEĞİL, anlamsal kapıdan (cache_get)
        geçer — böylece derlenmiş bilgi de güven eşiğine tabidir."""
        qtok = set(_tok(query))
        for r in sorted(self.data["rules"], key=lambda x: x["priority"]):
            if r["kind"] == "reflex" and r.get("response") and \
                    _trigger_hit(r.get("trigger"), qtok, query):
                return r
        return None

    def guard_patterns(self) -> list[str]:
        return [r["forbid"] for r in self.data["rules"]
                if r["kind"] == "guard" and r.get("forbid")]

    def triggered_tools(self, query: str) -> list[str]:
        qtok = set(_tok(query))
        return [t["name"] for t in self.data["tools"]
                if t["name"] in self.bound_tools
                and _trigger_hit(t.get("trigger"), qtok, t["name"])]

    def has_knowledge(self) -> bool:
        return bool(self.data["knowledge"])

    def search_knowledge(self, query: str, k: int = 3) -> list[str]:
        qtok = set(_tok(query))
        if not qtok:
            return []
        scored = []
        for row in self.data["knowledge"]:
            ktok = _tok(row["text"])
            overlap = sum(1 for t in ktok if t in qtok)
            if overlap:
                scored.append((overlap / (len(set(ktok)) ** 0.5), row["text"]))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [t for _, t in scored[:k]]

    def top_memory(self, n: int = 3) -> list[dict]:
        return sorted(self.data["memory"],
                      key=lambda m: (m["weight"], m["used"]), reverse=True)[:n]

    # ========= ANLAMSAL ÖNBELLEK + GÜVENLİ ÖĞRENME (derlenmiş zihin) =====
    def _ensure_index(self):
        """Tembel LSH indeksi: büyümede artımlı ekle, silme/embedder değişiminde
        yeniden kur. Küçük önbellekte indeks kullanılmaz (lineer zaten hızlı+kesin)."""
        cache = self.data["cache"]
        if len(cache) < INDEX_THRESHOLD:
            self._index = None
            return None
        from .index import LSHIndex
        cur = set(cache.keys())
        rebuild = (self._index is None or self._index_feature != self.embedder.id
                   or bool(self._indexed_keys - cur))  # silme olduysa yeniden kur
        if rebuild:
            idx = LSHIndex(self.embedder.dim)
            for k in cur:
                v = cache[k].get("vec")
                if v:
                    idx.add(k, v)
            self._index, self._indexed_keys = idx, cur
            self._index_feature = self.embedder.id
        else:
            for k in cur - self._indexed_keys:     # artımlı: yeni niyetleri ekle
                v = cache[k].get("vec")
                if v:
                    self._index.add(k, v)
            self._indexed_keys = cur
        return self._index

    def _best_intent(self, query: str):
        """Sorguya anlamca EN YAKIN derlenmiş niyeti bul (birebir cümle gerekmez).
        Büyük önbellekte LSH adayları + birebir-anahtar garantisi; küçükte lineer.
        LSH yaklaşık kaçırsa bile birebir/öğretilmiş eşleşme anahtarla bulunur."""
        qv = self.embedder.embed(query)
        cache = self.data["cache"]
        idx = self._ensure_index()
        if idx is not None:
            keys = idx.candidates(qv)
            ek = _key(query)
            if ek in cache:                        # birebir/öğretilmiş kesin garanti
                keys = set(keys); keys.add(ek)
            items = ((k, cache[k]) for k in keys)
        else:
            items = cache.items()
        best_k, best_c, best_s = None, None, -1.0
        for k, c in items:
            v = c.get("vec")
            if not v:
                continue
            s = cosine(qv, v)
            if s > best_s:
                best_k, best_c, best_s = k, c, s
        return best_k, best_c, best_s, qv

    def cache_confidence(self, query: str) -> float:
        _, _, s, _ = self._best_intent(query)
        return round(max(s, 0.0), 3)

    def cache_get(self, query: str) -> Optional[dict]:
        """Anlamsal getirme: güven eşiğini geçen, DOĞRULANMIŞ ve ENGELLENMEMİŞ
        derlenmiş cevabı döndür. Parafraz da tutar; alakasız sorgu tutmaz."""
        k, c, s, _ = self._best_intent(query)
        if c and s >= REUSE_SIM and c.get("verified") and not c.get("blocked"):
            c["_sim"] = round(s, 3)
            return c
        return None

    def cache_touch(self, query: str) -> None:
        """Yeniden kullanımda prototipi sorguya doğru çek (çevrimiçi öğrenme) ve
        tekrarı artır → kalıcı kurala derlenmeye yaklaş."""
        k, c, s, qv = self._best_intent(query)
        if c and s >= REUSE_SIM:
            c["hits"] = c.get("hits", 0) + 1
            # Öğretilmiş niyetin prototipi GROUND-TRUTH; gevşek eşleşmelerde
            # sürüklenip kirlenmemeli. Sadece keşfedilmiş niyetler uyarlanır.
            if c.get("source") != "taught":
                c["repeats"] = c.get("repeats", 0) + 1
                c["vec"] = mix_prototype(c.get("vec"), c.get("n", 1), qv)
                c["n"] = c.get("n", 1) + 1
                self._maybe_promote(k)

    def learn(self, query: str, answer: str, source: str = "model",
              verified: bool = True) -> Optional[str]:
        """Cevabı anlamsal niyet olarak derle. Aynı niyetin yeni ifadesiyse
        prototipi günceller; değilse yeni niyet açar. Güvenli: doğrulanmış değilse
        kurala yükseltilmez."""
        if not answer:
            return None
        k, c, s, qv = self._best_intent(query)
        # GÜVENLİK: engellenmiş niyet ne diriltilir ne de yeniden derlenir
        if c and s >= SAME_SIM and c.get("blocked"):
            return "engelli niyet — yeniden derleme reddedildi"
        exist = self.data["cache"].get(_key(query))
        if exist and exist.get("blocked"):
            return "engelli niyet — yeniden derleme reddedildi"
        if c and s >= SAME_SIM and c.get("source") == "taught":
            # Öğretilmiş niyet GROUND-TRUTH'tur: model çıktısı üzerine yazılmaz.
            return "öğretilmiş niyet korundu"
        if c and s >= SAME_SIM:
            c["answer"] = answer
            c["repeats"] = c.get("repeats", 0) + 1
            c["vec"] = mix_prototype(c.get("vec"), c.get("n", 1), qv)
            c["n"] = c.get("n", 1) + 1
            c["verified"] = bool(c.get("verified", True) and verified)
            key = k
        else:
            key = _key(query)
            self.data["cache"][key] = {
                "query": query, "answer": answer, "vec": qv, "n": 1,
                "hits": 0, "repeats": 1, "verified": verified, "reward": 0,
                "blocked": False, "learned": False, "source": source,
                "ts": time.time()}
        self.remember(f"S: {query}\nY: {answer}", kind="episodic")
        return self._maybe_promote(key)

    def _maybe_promote(self, key: str) -> Optional[str]:
        """GÜVENLİ DERLEME: yalnızca doğrulanmış, engellenmemiş, negatif-geri-
        beslemesiz ve yeterince tekrarlanmış niyet kalıcı 'learned' kurala derlenir.
        Böylece sistem YANLIŞ cevabı kurala derlemeyi reddeder."""
        c = self.data["cache"].get(key)
        if not c:
            return None
        if c.get("blocked") or not c.get("verified") or c.get("reward", 0) < 0:
            return "derleme reddedildi (doğrulanmadı / engellendi / negatif geri bildirim)"
        if c.get("repeats", 0) >= LEARN_THRESHOLD and not c.get("learned"):
            # Öğrenilmiş niyet ayrı anahtar-sözcük kuralı DEĞİL; bir bayrak.
            # Anlamsal kapıdan (cache_get) geçer, böylece güven eşiğine tabidir.
            c["learned"] = True
            return f"'{c['query'][:40]}' → kalıcı niyete derlendi"
        return f"önbellek (tekrar {c.get('repeats', 0)}/{LEARN_THRESHOLD})"

    def feedback(self, query: str, good: bool) -> str:
        """Geri besleme sinyali. Kötü → niyet engellenir ve derlenmiş kural SÖKÜLÜR
        (decompile); artık ne yeniden kullanılır ne de yeniden derlenir."""
        k, c, s, _ = self._best_intent(query)
        if not c or s < REUSE_SIM:
            return "eşleşen niyet yok"
        c["reward"] = c.get("reward", 0) + (1 if good else -1)
        if not good:
            c["blocked"] = True
            removed = self._decompile(c)
            self._maybe_save()
            return f"negatif geri bildirim → niyet engellendi, {removed} kural söküldü"
        self._maybe_save()
        return f"pozitif geri bildirim → ödül {c['reward']}"

    def _decompile(self, c: dict) -> int:
        """Derlenmiş niyeti söküp bayrağını indir (geri alınabilir öğrenme)."""
        was = 1 if c.get("learned") else 0
        c["learned"] = False
        return was

    # ================= İSTATİSTİK (maliyet modeli / ANALYZE) ============
    def observe_op(self, op: str, ms: float) -> None:
        d = self.data["stats"]["op_ms"]
        prev = d.get(op)
        d[op] = ms if prev is None else (EWMA_ALPHA * ms + (1 - EWMA_ALPHA) * prev)

    def op_latency(self, op: str) -> Optional[float]:
        return self.data["stats"]["op_ms"].get(op)

    def log_run(self, query: str, plan: list, trace: list) -> None:
        self.data["stats"]["runs"] += 1
        self.data["log"].append({"query": query, "plan": plan, "trace": trace,
                                 "ts": time.time()})
        self.data["log"] = self.data["log"][-200:]   # son 200 çalışma

    # ================= ÜST DÜZEY API =====================================
    def _planner(self) -> Planner:
        return Planner(self, self.engines, self.policy)

    def _vm(self) -> VM:
        return VM(self, self.engines, self.bound_tools, self.policy, self.ask_fn)

    def explain(self, query: str) -> Plan:
        """Seçilen planı döndür — EXPLAIN QUERY PLAN karşılığı."""
        return self._planner().plan(query)

    def run(self, query: str) -> Answer:
        plan = self._planner().plan(query)
        try:
            ans = self._vm().execute(plan)
        except SecurityError as e:
            # SQLite-seviyesi güvenilirlik: bütçe/güvenlik sınırı ÇÖKERTMEZ,
            # zarifçe durdurur.
            ans = Answer(f"(Durduruldu: {e})", plan,
                         [{"op": "ABORT", "error": str(e)}], Cost(),
                         source="abort", blocked=True)
        self._maybe_save()
        return ans

    def ask(self, query: str) -> str:
        return self.run(query).text

    def consolidate(self, max_intents: Optional[int] = None,
                    max_memory: int = 50, max_log: int = 50) -> dict:
        """Deneyimi sıkıştır: benzer niyetleri birleştir, uzun-kuyruk yararsızları
        buda, hafıza/günlüğü sınırla. Öğretilmiş/öğrenilmiş/engelli korunur →
        davranış ve güvenlik bozulmaz. Dosya trafikle doğrusal büyümez."""
        from . import konsolidasyon
        return konsolidasyon.consolidate(self, max_intents=max_intents,
                                         max_memory=max_memory, max_log=max_log)

    def set_embedder(self, embedder) -> dict:
        """Embedder'ı (opt-in güçlü/nöral/lexicon katmanı) değiştir ve TÜM
        vektörleri aynı uzayda yeniden hesapla. Uzay tutarlılığı korunur; bozulma
        olmaz. Güvenlik gateleri (eşik/VERIFY/feedback/öğretilmiş-koruma) aynen
        geçerli kalır."""
        self.embedder = embedder
        return self.reembed()

    def reembed(self) -> dict:
        n = 0
        for c in self.data["cache"].values():
            c["vec"] = self.embedder.embed(c.get("query", ""))
            c["n"] = 1
            n += 1
        self.data["feature"] = self.embedder.id
        self._maybe_save()
        return {"yeniden_gömülen": n, "feature": self.embedder.id}

    def sign(self, key: str, author: str = "") -> dict:
        """Beyni imzala (dağıtım öncesi kurcalama+kimlik koruması)."""
        from . import imza
        return imza.sign(self, key, author)

    def verify(self, key: str) -> dict:
        """İmzalı beyni doğrula: kurcalandı mı, kimden geldi?"""
        from . import imza
        return imza.verify(self, key)

    def stats(self) -> dict:
        d = self.data
        return {
            "format_version": d["mimar"],
            "role": d["role"],
            "rules": len(d["rules"]),
            "learned_rules": sum(1 for c in d["cache"].values() if c.get("learned")),
            "tools": len(d["tools"]),
            "knowledge": len(d["knowledge"]),
            "memory": len(d["memory"]),
            "cache": len(d["cache"]),
            "runs": d["stats"]["runs"],
            "op_ms": {k: round(v, 3) for k, v in d["stats"]["op_ms"].items()},
        }


def _looks_regex(s: str) -> bool:
    return any(ch in s for ch in r".*+?[](){}|\^$")
