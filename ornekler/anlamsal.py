"""
Mimar — Anlamsal seçici + güvenli derleme kanıtı.

Gösterir:
  1) Parafraz tutar: "birebir aynı soru" sorulmadan derlenmiş cevap getirilir.
  2) Alakasız sorgu TUTMAZ: yanlış niyet eşleşmesi olmaz (güven eşiği).
  3) Güvenli derleme: negatif geri bildirim alan cevap ENGELLENİR ve derlenmiş
     kural SÖKÜLÜR — sistem yanlışı tekrar sunmayı/derlemeyi reddeder.

Çevrimdışı (Echo) koşar.  python ornekler/anlamsal.py
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mimar
from mimar.engines import Echo


def ciz(b=""):
    print("\n" + "=" * 72)
    if b:
        print(b); print("=" * 72)


def sor(beyin, q):
    guven = beyin.cache_confidence(q)
    a = beyin.run(q)
    ops = [o["op"] for o in a.plan.as_list()]
    print(f"  «{q}»")
    print(f"     güven={guven:<5}  kaynak={a.source:<6}  ops={ops}")
    return a


beyin = mimar.open(None, engines=[Echo()])   # bellekte
beyin.declare(role="Acme Türkçe destek asistanı.")

ciz("1) ÖĞRET — niyeti ilk kez modelle çöz (bir kez)")
sor(beyin, "kargom ne zaman gelir")

ciz("2) PARAFRAZ — FARKLI kelimelerle AYNI niyet → derlenmiş cevap gelmeli")
print("(birebir aynı cümle YOK; anlamsal eşleşme bekleniyor)")
sor(beyin, "kargo ne zaman gelecek acaba")
sor(beyin, "siparişimin kargosu ne zaman elime ulaşır")
sor(beyin, "paketim kaç günde gelir")

ciz("3) ALAKASIZ — farklı niyet → eşleşmemeli, modele gitmeli")
sor(beyin, "iade işlemini nasıl yaparım")
sor(beyin, "mağaza çalışma saatleriniz nedir")

ciz("4) GÜVENLİ DERLEME — yanlış cevaba NEGATİF geri bildirim")
beyin2 = mimar.open(None, engines=[Echo()])
beyin2.declare(role="test")
print("Aynı niyeti iki kez sor → tekrar eşiğini geçip kalıcı niyete derlensin:")
sor(beyin2, "şifremi nasıl sıfırlarım")   # 1: model + önbellek (tekrar 1)
sor(beyin2, "şifremi nasıl sıfırlarım")   # 2: önbellek + tekrar 2 → derlenir
print("  Derlenmiş niyet sayısı:", beyin2.stats()["learned_rules"])

print("\n  >> Kullanıcı: bu cevap YANLIŞ (negatif geri bildirim)")
print("   ", beyin2.feedback("şifremi nasıl sıfırlarım", good=False))
print("  Derlenmiş niyet sayısı (sökülmeli → 0):", beyin2.stats()["learned_rules"])

print("\n  Aynı niyeti tekrar sor — artık ENGELLİ, derlenmiş gelmemeli, modele gitmeli:")
sor(beyin2, "şifremi nasıl sıfırlarım")
print("  Zorla yeniden derlemeyi dene:",
      "->", beyin2.learn("şifremi nasıl sıfırlarım", "yeni cevap"))

ciz("5) GÜVEN SPEKTRUMU — eşik (REUSE_SIM) gerçekten ayırıyor mu?")
from mimar.brain import REUSE_SIM, SAME_SIM
print(f"  REUSE_SIM={REUSE_SIM}  SAME_SIM={SAME_SIM}")
for q in ["kargom ne zaman gelir", "kargo ne zaman gelecek",
          "paketim kaç günde gelir", "iade nasıl yapılır",
          "hava bugün nasıl"]:
    print(f"    güven={beyin.cache_confidence(q):<6} «{q}»")

ciz("BİTTİ")
