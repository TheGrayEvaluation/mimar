"""
Mimar — Çekirdek test takımı (invariant'lar).

"Çalışan kısımlardan emin olmak" için: demolar değil, DEĞİŞMEZ KURALLAR.
Teorik çerçeve her testin başında belirtilir. Bağımlılıksız koşar:

    python testler/test_cekirdek.py

Her test bir invariant'ı savunur; biri bozulursa çekirdek güvenilmez demektir.
"""

import math
import os
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mimar
from mimar.brain import REUSE_SIM, SAME_SIM
from mimar.engines import Completion, Echo
from mimar.selector import Embedder, cosine
from mimar.security import Policy

_OK = 0
_FAIL = 0


def check(cond, msg):
    global _OK, _FAIL
    if cond:
        _OK += 1
        print(f"  ✓ {msg}")
    else:
        _FAIL += 1
        print(f"  ✗ BAŞARISIZ: {msg}")


class StrongEngine(Echo):
    """Test için güçlü motor taklidi (maliyet modeli seçimini sınamak)."""
    name = "strong"
    tier = "frontier"
    strength = 0.95
    usd_per_call = 0.002
    latency_ms = 1500.0


# ---------------------------------------------------------------------------
def test_embedder():
    """TEORİ: embedding L2-normalize → cosine ∈ [-1,1]; özdeşlik=1. Nearest-
    centroid'in geometrisi buna dayanır (Voronoi/marj)."""
    print("\n[1] Embedder geometrisi")
    e = Embedder()
    v = e.embed("kargom ne zaman gelir")
    norm = math.sqrt(sum(x * x for x in v))
    check(abs(norm - 1.0) < 1e-6, f"vektör birim normlu (|v|={norm:.6f})")
    check(len(v) == e.dim, f"boyut sabit ({e.dim})")
    check(abs(cosine(v, v) - 1.0) < 1e-6, "cosine(v,v)=1 (özdeşlik)")
    z = e.embed("")
    check(all(x == 0.0 for x in z), "boş metin → sıfır vektör (çökme yok)")
    check(cosine(v, z) == 0.0, "sıfır vektörle cosine=0 (güvenli)")
    # Parafraz > alakasız (sıralama invariant'ı)
    para = e.embed("kargo ne zaman gelecek")
    alaka = e.embed("hava bugün nasıl")
    check(cosine(v, para) > cosine(v, alaka),
          f"parafraz({cosine(v,para):.3f}) > alakasız({cosine(v,alaka):.3f})")


def test_format_roundtrip():
    """TEORİ: kalıcılık = dosya↔bellek izomorfizmi. Yaz-oku-aç aynı durumu
    vermeli; aksi halde 'taşınabilir beyin' iddiası çöker."""
    print("\n[2] Öz format yaz/oku turu")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.mimar")
        b = mimar.open(path, engines=[Echo()])
        b.declare(role="Test rolü", knowledge=["bilgi bir"],
                  rules=[{"text": "selam", "kind": "reflex",
                          "trigger": "selam", "response": "merhaba"}])
        b.run("yeni bir soru bu")       # cache + stats üret
        b.close()

        # Magic imza
        with open(path, "rb") as f:
            head = f.read(4)
        check(head == b"MIMR", "dosya magic imzası 'MIMR'")

        b2 = mimar.open(path, engines=[Echo()])
        s = b2.stats()
        check(s["role"] == "Test rolü", "rol korundu")
        check(s["knowledge"] == 1, "bilgi korundu")
        check(s["rules"] >= 1, "kurallar korundu")
        check(s["runs"] == 1, "çalışma istatistiği korundu")
        check(len(b2.data["cache"]) == 1, "önbellek (vektörüyle) korundu")
        # Vektör gerçekten float listesi olarak döndü mü?
        vec = list(b2.data["cache"].values())[0].get("vec")
        check(isinstance(vec, list) and len(vec) == Embedder.dim,
              "önbellek vektörü JSON turunu sağ geçti")


def test_magic_reject():
    """TEORİ: tip güvenliği — rastgele bayt bir beyin değildir."""
    print("\n[3] Yabancı dosya reddi")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sahte.mimar")
        with open(path, "wb") as f:
            f.write(b"bu bir mimar dosyasi degil")
        try:
            mimar.open(path, engines=[Echo()])
            check(False, "yabancı dosya reddedilmeliydi")
        except ValueError:
            check(True, "yabancı dosya ValueError ile reddedildi")


def test_novel_then_compile():
    """TEORİ: 'kullanıldıkça derlenme' — yeni niyet model ister; tekrar eşiğini
    (LEARN_THRESHOLD) geçince kalıcı niyete derlenir."""
    print("\n[4] Yeni → model, tekrar → derlenir")
    b = mimar.open(None, engines=[Echo()])
    b.declare(role="t")
    a1 = b.run("fatura adresimi nasıl değiştiririm")
    check(a1.source == "model", "yeni niyet modele gitti")
    a2 = b.run("fatura adresimi nasıl değiştiririm")
    check(a2.source in ("cache", "rule"), "tekrar derlenmiş yoldan geldi")
    check(b.stats()["learned_rules"] == 1, "kalıcı niyete derlendi (learned=1)")
    check(a2.cost.ms <= a1.cost.ms + 1e-6 or a2.source != "model",
          "derlenmiş yol modelden pahalı değil")


def test_semantic_reuse_and_abstention():
    """TEORİ: selective prediction / abstention — güven ≥ eşik ise yeniden kullan,
    değilse ÇEKİL (modele git). Asla sahte eşleşme üretme (açık-küme tanıma)."""
    print("\n[5] Anlamsal yeniden kullanım + çekilme (abstention)")
    b = mimar.open(None, engines=[Echo()])
    b.declare(role="t")
    b.run("kargom ne zaman gelir")
    # Morfolojik parafraz: eşik üstü beklenir
    conf_para = b.cache_confidence("kargo ne zaman gelecek acaba")
    a_para = b.run("kargo ne zaman gelecek acaba")
    check(conf_para >= REUSE_SIM and a_para.source in ("cache", "rule"),
          f"parafraz yeniden kullanıldı (güven={conf_para:.3f} ≥ {REUSE_SIM})")
    # Alakasız: eşik altı → model (çekilme)
    conf_alaka = b.cache_confidence("iade işlemi nasıl yapılır")
    a_alaka = b.run("iade işlemi nasıl yapılır")
    check(conf_alaka < REUSE_SIM and a_alaka.source == "model",
          f"alakasız sorgu çekildi/modele gitti (güven={conf_alaka:.3f} < {REUSE_SIM})")


def test_safe_compilation():
    """TEORİ: geri-alınabilir öğrenme — negatif geri bildirim niyeti engeller,
    derlenmişi söker (decompile) ve yeniden derlemeyi reddeder. Sistem bir hatayı
    kalıcılaştırmaz."""
    print("\n[6] Güvenli derleme (yanlışı reddet/geri al)")
    b = mimar.open(None, engines=[Echo()])
    b.declare(role="t")
    b.run("şifremi nasıl sıfırlarım")
    b.run("şifremi nasıl sıfırlarım")    # derlenir
    check(b.stats()["learned_rules"] == 1, "önce derlendi")
    msg = b.feedback("şifremi nasıl sıfırlarım", good=False)
    check(b.stats()["learned_rules"] == 0, "negatif geri bildirim → söküldü")
    a = b.run("şifremi nasıl sıfırlarım")
    check(a.source == "model", "engelli niyet artık modele gidiyor")
    r = b.learn("şifremi nasıl sıfırlarım", "yeni cevap")
    check("reddedildi" in (r or ""), "engelli niyet yeniden derlenmedi")


def test_guard_block():
    """TEORİ: çıktı sözleşmesi — guard deseni ihlal eden yanıt yayınlanmaz."""
    print("\n[7] Guard (yasak desen) engellemesi")
    b = mimar.open(None, engines=[Echo(prefix="iade sözü veriyorum")])
    b.declare(role="t", guards=["iade sözü"])
    a = b.run("herhangi bir soru")
    check(a.blocked is True, "guard ihlali engellendi")
    check("güvenlik" in a.text.lower() or "engellendi" in a.text.lower(),
          "engel mesajı döndü")


def test_budget_graceful():
    """TEORİ: kaynak sınırı (bütçe) bir güvenlik değişmezidir; aşım ÇÖKERTMEZ,
    zarifçe durdurur (SQLite-seviyesi güvenilirlik)."""
    print("\n[8] Bütçe aşımı zarif durdurma")
    pol = Policy(max_model_calls=0)
    b = mimar.open(None, engines=[Echo()], policy=pol)
    b.declare(role="t")
    a = b.run("model gerektiren yeni soru")   # model çağrısı bütçesi 0
    check(a.blocked is True and a.source == "abort",
          "bütçe aşımı çökmeden engellendi")


def test_tool_allowlist():
    """TEORİ: en az yetki — araç hem BAĞLI hem İZİNLİ olmalı."""
    print("\n[9] Araç izin listesi (en az yetki)")
    calls = {"n": 0}
    b = mimar.open(None, engines=[Echo()])
    b.declare(role="t")

    @b.tool(trigger="bakiye", description="bakiye sorgular")
    def bakiye(q):
        calls["n"] += 1
        return "100 TL"

    b.run("bakiye ne kadar")
    check(calls["n"] == 1, "bağlı+izinli araç çağrıldı")
    # Şimdi izin listesini kısıtla: aracı dışla
    b.policy.allow_tools = set()           # hiçbir araca izin yok
    b.run("bakiye ne kadar tekrar")
    check(calls["n"] == 1, "izinsiz araç çağrılmadı (en az yetki)")


def test_risk_engine_choice():
    """TEORİ: maliyet modeli — risk=normal → en ucuz motor; risk=high → en güçlü
    motor + zorunlu doğrulama."""
    print("\n[10] Risk'e göre motor seçimi")
    b = mimar.open(None, engines=[Echo(), StrongEngine()])
    b.declare(role="t")
    b.policy.risk = "normal"
    p = b.explain("bir soru")
    tier_normal = [o.args.get("engine") for o in p.ops if o.name == "MODEL"]
    check(tier_normal == ["mock"], f"risk=normal ucuz motor ({tier_normal})")
    b.policy.risk = "high"
    b.policy.min_strength = 0.9
    p2 = b.explain("kritik bir soru")
    tier_high = [o.args.get("engine") for o in p2.ops if o.name == "MODEL"]
    has_verify = any(o.name == "VERIFY" for o in p2.ops)
    check(tier_high == ["frontier"], f"risk=high güçlü motor ({tier_high})")
    check(has_verify, "risk=high doğrulama içeriyor")


def test_explain_inspectable():
    """TEORİ: şeffaflık — her karar bir plandır, EXPLAIN gibi incelenebilir."""
    print("\n[11] Plan incelenebilirliği (EXPLAIN)")
    b = mimar.open(None, engines=[Echo()])
    b.declare(role="t")
    p = b.explain("açıklanabilir mi")
    check(len(p.ops) >= 2, "plan en az iki operatör içeriyor")
    check(any(o.name == "VERIFY" for o in p.ops), "plan doğrulama ile bitiyor")
    check(p.est.ms >= 0, "plan maliyet tahmini taşıyor")


def test_prototype_learning():
    """TEORİ: çevrimiçi merkezcil güncelleme — yeniden kullanım prototipi örneğe
    çeker (kanıt biriktikçe eş-ifadeler yakınsar)."""
    print("\n[12] Prototip çevrimiçi öğrenme")
    b = mimar.open(None, engines=[Echo()])
    b.declare(role="t")
    b.run("kargom ne zaman gelir")
    intent = list(b.data["cache"].values())[0]
    n0 = intent["n"]
    b.run("kargo ne zaman gelecek acaba")   # parafraz yeniden kullanır → prototip güncellenir
    check(intent["n"] > n0, f"prototip örnek sayısı arttı ({n0}→{intent['n']})")


def test_behavior_distillation():
    """TEORİ: davranış klonlama / bilgi damıtma — yakalanan girdi→çıktı çiftleri
    taşınabilir dosyaya damıtılır; davranış motordan BAĞIMSIZ yeniden üretilir ve
    eval ile ölçülür (iddia değil, ölçüm)."""
    print("\n[13] Davranış damıtma (taşınabilir + ölçülür)")
    from mimar import davranis

    class Target(Echo):
        name = "hedef"
        strength = 0.9

        def complete(self, messages, **o):
            u = next((m.content for m in reversed(messages)
                      if m.role == "user"), "")
            return Completion(text=f"HEDEF[{u.strip()}]", engine=self.name)

    prompts = ["soru bir", "ikinci konu", "bambaşka bir istek", "dördüncü mesele"]
    pairs = davranis.capture(Target(), prompts)
    b = mimar.open(None, engines=[Echo()])
    b.declare(role="t")
    davranis.ingest(b, pairs)
    raw = davranis.raw_fidelity(Echo(), b.system_prompt(), pairs, b.embedder)
    wb = davranis.fidelity(b, Echo(), pairs)
    wb2 = davranis.fidelity(b, Echo(prefix="X"), pairs)
    check(raw["oran"] == 0.0, f"beyin yokken davranış yok (raw={raw['oran']})")
    check(wb["oran"] == 1.0, f"beyin varken davranış tam (wb={wb['oran']})")
    check(wb2["oran"] == 1.0, f"farklı motorda da tam → taşınabilir (wb2={wb2['oran']})")


def test_merge():
    """TEORİ: birleştirilebilirlik — iki bağımsız beyin, provenanslı tek beyne
    birleşir; birleşik beyin her iki alanı bilir, çakışmalar politikayla çözülüp
    raporlanır, guard'lar toplamsaldır (sessiz ezme yok)."""
    print("\n[14] Beyin birleştirme (MERGE + provenans)")
    from mimar import merge
    a = mimar.open(None, engines=[Echo()])
    a.declare(role="A", guards=["yasak-a"])
    a.teach("alfa sorusu", "alfa cevabı")
    a.teach("ortak soru", "A'nın cevabı")
    a.feedback("ortak soru", good=True)      # ödül avantajı
    b = mimar.open(None, engines=[Echo()])
    b.declare(role="B", guards=["yasak-b"])
    b.teach("beta sorusu", "beta cevabı")
    b.teach("ortak soru", "B'nin cevabı")    # çakışma

    m, rapor = merge.merge_new(None, [a, b], engines=[Echo()])
    check(len(m.data["cache"]) == 3, "niyetler birleşti (3 benzersiz)")
    check(len(rapor["sources"]) == 2 and all(p["hash"] for p in rapor["sources"]),
          "iki kaynak provenansı (hash'li) kaydedildi")
    check(any(c["tür"] == "niyet" for c in rapor["conflicts"]),
          "çakışma tespit edilip raporlandı")
    check(set(m.guard_patterns()) == {"yasak-a", "yasak-b"},
          "guard'lar toplamsal (iki beyninki de)")
    m.engines = [Echo()]
    check(m.run("alfa sorusu").source in ("rule", "cache") and
          m.run("beta sorusu").source in ("rule", "cache"),
          "birleşik beyin her iki alanı da cevaplıyor")
    check(m.run("ortak soru").text == "A'nın cevabı",
          "çakışma ödül avantajlı kaynağa (A) çözüldü")


def test_mcp():
    """TEORİ: birlikte çalışabilirlik — beyin standart bir MCP sunucusu olarak
    açılır; herhangi bir MCP ajanı kodunu değiştirmeden mount eder."""
    print("\n[15] MCP montajı (JSON-RPC sözleşmesi)")
    from mimar.mcp_server import MimarMCP
    b = mimar.open(None, engines=[Echo()])
    b.declare(role="t")
    b.teach("x nedir", "x cevabı")
    srv = MimarMCP(b)
    init = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    check(init["result"]["serverInfo"]["name"] == "mimar", "initialize el sıkışması")
    check("protocolVersion" in init["result"], "protokol sürümü bildirildi")
    tl = srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = [t["name"] for t in tl["result"]["tools"]]
    check("mimar_ask" in names and "mimar_teach" in names, "araçlar listeleniyor")
    note = srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"})
    check(note is None, "bildirime yanıt verilmez (JSON-RPC)")
    call = srv.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                       "params": {"name": "mimar_ask",
                                  "arguments": {"query": "x nedir"}}})
    txt = call["result"]["content"][0]["text"]
    check("x cevabı" in txt and call["result"]["isError"] is False,
          "mimar_ask öğretilen cevabı döndürdü")
    err = srv.handle({"jsonrpc": "2.0", "id": 4, "method": "yok"})
    check(err.get("error", {}).get("code") == -32601, "bilinmeyen yöntem → -32601")


def test_consolidation():
    """TEORİ: sıkıştırma — ağır kullanımda beyin ŞİŞMEMELİ; benzerler birleşir,
    uzun-kuyruk budanır, hafıza/günlük sınırlanır; ama öğretilmiş davranış KORUNUR
    (doğruluk bozulmaz). Deneyim satır değil, öz olarak yaşar."""
    print("\n[16] Konsolidasyon (şişmeme + doğruluk korunur)")
    import random
    random.seed(3)
    kanon = {"alfa kanon soru": "alfa", "beta kanon soru": "beta",
             "gama kanon soru": "gama"}
    b = mimar.open(None, engines=[Echo()])
    b.declare(role="t")
    for q, a in kanon.items():
        b.teach(q, a)
    pool = ["masa", "kalem", "deniz", "uçak", "kitap", "bulut", "orman", "nehir",
            "dağ", "şehir", "köprü", "bahçe", "pencere", "duvar", "çatı", "yıldız",
            "güneş", "toprak", "rüzgar", "yağmur", "metal", "ahşap", "iplik",
            "kumaş", "deri", "çelik", "altın", "gümüş", "mermer", "kömür", "petrol",
            "ışık", "gölge", "koku", "sayı", "harf", "balık", "kuş", "böcek", "çiçek"]
    for _ in range(40):
        b.run(" ".join(random.sample(pool, 3)))
    before = len(b.data["cache"])
    rep = b.consolidate(max_intents=8, max_memory=5)
    after = len(b.data["cache"])
    check(after < before, f"niyet sayısı küçüldü ({before}→{after})")
    check(len(b.data["memory"]) <= 5, "hafıza üst sınıra çekildi")
    dogru = sum(1 for q, a in kanon.items() if b.run(q).text == a)
    check(dogru == 3, f"öğretilmiş davranış korundu ({dogru}/3)")
    check(all(b.run(q).source in ("rule", "cache") for q in kanon),
          "kanonik niyetler hâlâ derlenmiş yoldan geliyor")


def test_pluggable_embedder():
    """TEORİ: embedder takılabilir (nöral/özel katman buraya takılır) AMA uzay
    tutarlılığı korunur. Eş-anlam SÖZLÜĞÜ YOK (kırılgan): derin eş-anlam ya nöral
    embedder'dan gelir ya da güvenle modele düşülür. Burada arayüz + güvenlik
    sınanır (bağımlılıksız mock 'nöral' embedder ile)."""
    print("\n[17] Takılabilir embedder arayüzü + uzay tutarlılığı")
    import hashlib
    from mimar.selector import FunctionEmbedder

    def mock_encode(text):   # deterministik, FARKLI uzay (dim 16) — nöral taklidi
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [bt / 255.0 for bt in h[:16]]

    emb = FunctionEmbedder(mock_encode, dim=16, id="mock-neural")
    b = mimar.open(None, engines=[Echo()], embedder=emb)
    b.declare(role="t")
    b.teach("gizli soru", "gizli cevap")
    a = b.run("gizli soru")
    check(a.text == "gizli cevap" and a.source in ("rule", "cache"),
          "özel embedder altında birebir reuse çalışıyor")
    # Uzay tutarlılığı: farklı embedder'la açış sessizce bozmaz, net hata verir
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "f.mimar")
        bb = mimar.open(p, engines=[Echo()],
                        embedder=FunctionEmbedder(mock_encode, 16, "mock-neural"))
        bb.teach("x", "y"); bb.close()
        try:
            mimar.open(p, engines=[Echo()])   # varsayılan hash ≠ mock-neural
            check(False, "uyumsuz embedder açılışı hata vermeliydi")
        except ValueError:
            check(True, "uyumsuz embedder açılışı net hata (sessiz bozma yok)")


def test_index_exact_guarantee():
    """TEORİ: yaklaşık indeks (LSH) birebir/öğretilmiş eşleşmeyi ASLA kaçırmamalı.
    Anahtar-tabanlı kesin bulma garantisi: büyük önbellekte bile verbatim reuse
    %100 doğru; LSH yalnız parafraz adaylarını hızlandırır (kaçırırsa modele düşülür,
    yanlış cevap değil)."""
    print("\n[18] İndeks: birebir/öğretilmiş kesin bulma garantisi")
    import random
    rng = random.Random(11)
    harf = "abcdefghijklmnoprstuvyz"

    def kelime():
        return "".join(rng.choice(harf) for _ in range(rng.randint(4, 7)))

    b = mimar.open(None, engines=[Echo()])
    b.declare(role="t")
    cift = {}
    for i in range(80):
        q = " ".join(kelime() for _ in range(4))
        cift[q] = f"cevap-{i}"
        b.teach(q, cift[q])
    check(len(b.data["cache"]) >= 64, f"önbellek indeks eşiğini aştı ({len(b.data['cache'])})")
    check(b._ensure_index() is not None, "LSH indeksi devrede")
    dogru = sum(1 for q, a in cift.items() if b.run(q).text == a)
    check(dogru == len(cift),
          f"indeks altında verbatim reuse %100 ({dogru}/{len(cift)})")


def test_signing():
    """TEORİ: dağıtım güveni — imza içerik-hash'i üzerinden kurcalamayı tespit
    eder ve kaynağı doğrular; yanlış anahtar/değişen içerik reddedilir; imza
    diske yaz-oku turunu sağ geçer."""
    print("\n[19] İmzalama (kurcalama + kimlik)")
    import os
    import tempfile
    b = mimar.open(None, engines=[Echo()])
    b.declare(role="dağıtım")
    b.teach("soru", "cevap")
    b.sign("anahtar", author="ufuk")
    check(b.verify("anahtar")["ok"] is True, "doğru anahtar → geçerli")
    check(b.verify("yanlis")["ok"] is False, "yanlış anahtar → reddedildi")
    # Kurcalama: içeriği değiştir → doğrulama bozulmalı
    list(b.data["cache"].values())[0]["answer"] = "kurcalanmış"
    v = b.verify("anahtar")
    check(v["ok"] is False and "kurcalama" in v["neden"],
          "içerik değişti → kurcalama tespit edildi")
    # Round-trip: imzala, kaydet, yeniden aç, doğrula
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.mimar")
        bb = mimar.open(p, engines=[Echo()])
        bb.declare(role="r"); bb.teach("a", "b"); bb.sign("k", author="z")
        bb.close()
        b2 = mimar.open(p, engines=[Echo()])
        check(b2.verify("k")["ok"] is True, "imza yaz-oku turunu sağ geçti")


def main():
    tests = [test_embedder, test_format_roundtrip, test_magic_reject,
             test_novel_then_compile, test_semantic_reuse_and_abstention,
             test_safe_compilation, test_guard_block, test_budget_graceful,
             test_tool_allowlist, test_risk_engine_choice,
             test_explain_inspectable, test_prototype_learning,
             test_behavior_distillation, test_merge, test_mcp,
             test_consolidation, test_pluggable_embedder,
             test_index_exact_guarantee, test_signing]
    print("=" * 60)
    print("MİMAR ÇEKİRDEK TEST TAKIMI")
    print("=" * 60)
    for t in tests:
        try:
            t()
        except Exception as e:
            global _FAIL
            _FAIL += 1
            print(f"  ✗ İSTİSNA {t.__name__}: {e}")
    print("\n" + "=" * 60)
    print(f"SONUÇ: {_OK} geçti, {_FAIL} başarısız")
    print("=" * 60)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
