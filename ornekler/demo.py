"""
Mimar demosu — çekirdeğin çalıştığını ve "kullanıldıkça derlenme" tezini kanıtlar.

Hiçbir API anahtarı / indirme gerekmez: çevrimdışı Echo motoruyla koşar.
Çalıştır:  python ornekler/demo.py
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")       # Windows konsolu için
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mimar
from mimar.engines import Echo, LocalEngine, OpenAIEngine


def cizgi(baslik=""):
    print("\n" + "=" * 70)
    if baslik:
        print(baslik)
        print("=" * 70)


DOSYA = os.path.join(os.path.dirname(__file__), "destek.mimar")
if os.path.exists(DOSYA):
    os.remove(DOSYA)

# --- 1) NİYET BİLDİRİMİ ------------------------------------------------
cizgi("1) Geliştirici NİYETİ bildirir (rol + bilgi + kural + araç)")

beyin = mimar.open(DOSYA, engines=[Echo()])
beyin.declare(
    role="Acme mağazası için Türkçe müşteri destek asistanı.",
    guards=["iade sözü ver", "ücretsiz"],            # çıktıda geçmesi yasak
    knowledge=[
        "Kargo siparişten sonra 2-3 iş gününde teslim edilir.",
        "Çalışma saatlerimiz hafta içi 09:00-18:00 arasıdır.",
    ],
    rules=[
        {"text": "selam", "kind": "reflex", "trigger": "selam|merhaba",
         "response": "Merhaba! Size nasıl yardımcı olabilirim?"},
    ],
)


@beyin.tool(trigger="sipariş|kargo|nerede", description="Sipariş durumunu sorgular")
def siparis_durumu(sorgu: str) -> str:
    return "Sipariş #123: kargoya verildi, yarın teslim."


print(beyin.stats())

# --- 2) EXPLAIN: planı göster (çalıştırmadan) --------------------------
cizgi("2) EXPLAIN — reflex kuralı eşleşen soru için plan (model YOK)")
print(beyin.explain("merhaba"))

cizgi("2b) EXPLAIN — bilgi + araç gerektiren yeni soru için tam hat")
print(beyin.explain("kargom nerede acaba"))

# --- 3) İLK ÇALIŞMA: pahalı yol (MODEL) --------------------------------
cizgi("3) Soruyu İLK kez sor — optimizasyoncu tam hattı (MODEL) seçer")
soru = "kargom ne zaman gelir"
c1 = beyin.run(soru)
print("Plan kaynağı :", c1.source)
print("Maliyet      :", c1.cost.as_dict())
print("Operatörler  :", [o["op"] for o in c1.plan.as_list()])
print("Cevap        :", c1.text.replace("\n", " | "))

# İkinci kez: cache'e yazıldı; üçüncüde LEARN eşiği (2) dolup kurala derlenecek
beyin.run(soru)

# --- 4) AYNI SORU TEKRAR: ucuz yol (derlenmiş) -------------------------
cizgi("4) AYNI soruyu TEKRAR sor — optimizasyoncu artık derlenmiş yolu seçer")
c2 = beyin.run(soru)
print("Plan kaynağı :", c2.source, "  <-- model yerine önceden derlenmiş")
print("Maliyet      :", c2.cost.as_dict())
print("Operatörler  :", [o["op"] for o in c2.plan.as_list()])
print("Cevap        :", c2.text.replace("\n", " | "))

hizlanma = (c1.cost.ms / c2.cost.ms) if c2.cost.ms else float("inf")
print(f"\n>>> KANIT: ilk {c1.cost.ms:.2f}ms -> tekrar {c2.cost.ms:.2f}ms "
      f"(~{hizlanma:.0f}x hızlı), model çağrısı: {('VAR' if c1.source=='model' else 'yok')}"
      f" -> {('yok' if c2.source!='model' else 'VAR')}")
print(">>> Beyin kullanıldıkça kendini DERLEDİ: aynı iş artık bedava + anında.")

# --- 5) GÜVENLİK: guard ihlalini engelle -------------------------------
cizgi("5) YERLEŞİK GÜVENLİK — guard kuralını ihlal eden çıktı engellenir")
beyin2 = mimar.open(None, engines=[Echo(prefix="iade sözü ver")])  # kasıtlı ihlal
beyin2.declare(role="test", guards=["iade sözü ver"])
c3 = beyin2.run("bir şey sor")
print("Engellendi mi:", c3.blocked)
print("Cevap        :", c3.text)

# --- 6) MODEL BAĞIMSIZLIĞI + RİSK ile motor seçimi ---------------------
cizgi("6) MODEL BAĞIMSIZ — risk'e göre motor seçimi (maliyet modeli)")
cok_motor = mimar.open(None, engines=[Echo(), LocalEngine(), OpenAIEngine()])
cok_motor.declare(role="test")
p_ucuz = cok_motor.explain("normal bir soru")
print("risk=normal  -> seçilen motor katmanı:",
      [o.args.get("engine") for o in p_ucuz.ops if o.name == "MODEL"])
cok_motor.policy.risk = "high"
cok_motor.policy.min_strength = 0.9
p_guclu = cok_motor.explain("riskli, kritik bir soru")
print("risk=high    -> seçilen motor katmanı:",
      [o.args.get("engine") for o in p_guclu.ops if o.name == "MODEL"],
      "(güçlü motor + zorunlu doğrulama)")

# --- 7) TAŞINABİLİRLİK: dosyayı kapat, yeniden aç ----------------------
cizgi("7) TAŞINABİLİRLİK — beyin diske derlendi, kapat & yeniden aç")
beyin.close()
boyut = os.path.getsize(DOSYA)
beyin3 = mimar.open(DOSYA, engines=[Echo()])
print(f"Dosya        : {os.path.basename(DOSYA)}  ({boyut} bayt)")
print("Yeniden açılan beynin istatistiği:", beyin3.stats())
print("Öğrenilmiş kural sayısı:", beyin3.stats()["learned_rules"],
      "(deneyim dosyaya derlendi)")

cizgi("BİTTİ")
