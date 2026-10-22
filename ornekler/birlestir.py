"""
Mimar — Beyin birleştirme (MERGE) kanıtı.

İki ayrı alan beyni (kargo + iade) → provenanslı tek beyin. Gösterir:
  - Birleşik beyin HER İKİ alanı da bilir.
  - Her kaynak imzalı provenansla girer (kim, hangi içerik-hash'i).
  - Çakışma (aynı sorgu, farklı cevap) politikayla çözülür ve raporlanır.
  - Guard'lar toplamsal (iki beynin yasakları da geçerli).
  - Dosyaya derlenir, yeniden açılır.

python ornekler/birlestir.py
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mimar
from mimar import merge
from mimar.engines import Echo


def ciz(b=""):
    print("\n" + "=" * 70)
    if b:
        print(b); print("=" * 70)


# --- Kaynak 1: KARGO alanı beyni ---
kargo = mimar.open(None, engines=[Echo()])
kargo.declare(role="Kargo destek uzmanı.", guards=["kargo bedava"])
kargo.teach("kargom ne zaman gelir", "Kargonuz 2-3 iş gününde teslim edilir.")
kargo.teach("kargo takip nasıl yapılır", "Takip numaranızı sitemizden girin.")
kargo.teach("merhaba", "Kargo destek hattına hoş geldiniz.")   # ÇAKIŞMA adayı
kargo.feedback("merhaba", good=True)    # ödülü yükselt → çakışmada avantaj

# --- Kaynak 2: İADE alanı beyni ---
iade = mimar.open(None, engines=[Echo()])
iade.declare(role="İade & geri ödeme uzmanı.", guards=["iade sözü garanti"])
iade.teach("ürünü nasıl iade ederim", "14 gün içinde ücretsiz iade hakkınız var.")
iade.teach("param ne zaman yatar", "İade onayından sonra 5 iş günü içinde.")
iade.teach("merhaba", "İade departmanına hoş geldiniz.")       # ÇAKIŞMA adayı


@iade.tool(trigger="iade başlat", description="İade talebi açar")
def iade_baslat(q):
    return "İade #R-55 açıldı."


ciz("1) İKİ AYRI ALAN BEYNİ")
print("  Kargo beyni:", kargo.stats()["cache"], "niyet,",
      kargo.stats()["rules"], "kural")
print("  İade  beyni:", iade.stats()["cache"], "niyet,",
      iade.stats()["tools"], "araç")

ciz("2) MERGE — provenanslı tek beyne derle")
with __import__("tempfile").TemporaryDirectory() as d:
    yol = os.path.join(d, "birlesik.mimar")
    birlesik, rapor = merge.merge_new(yol, [kargo, iade], engines=[Echo()])

    print("  Birleşik:", rapor["merged"])
    print("\n  PROVENANS (kim, hangi hash):")
    for p in rapor["sources"]:
        print(f"    {p['tag']}: {p['source']}  hash={p['hash']}  "
              f"({p['intents']} niyet, {p['rules']} kural)")

    print("\n  ÇAKIŞMALAR (sessiz ezme yok):")
    for c in rapor["conflicts"]:
        print(f"    {c}")

    ciz("3) BİRLEŞİK BEYİN HER İKİ ALANI DA BİLİR")
    birlesik.engines = [Echo()]
    for q in ["kargom ne zaman gelir", "ürünü nasıl iade ederim",
              "param ne zaman yatar", "kargo takip nasıl yapılır"]:
        a = birlesik.run(q)
        print(f"    [{a.source}] «{q}» -> {a.text}")

    ciz("4) ÇAKIŞAN 'merhaba' — politika kazananı seçti")
    a = birlesik.run("merhaba")
    key = mimar.brain._key("merhaba")
    src = birlesik.data["cache"][key].get("_src")
    print(f"    «merhaba» -> {a.text}   (kazanan kaynak: {src}, ödül avantajı)")

    ciz("5) GUARD'LAR TOPLAMSAL (iki beynin yasakları da geçerli)")
    print("    Guard desenleri:", birlesik.guard_patterns())

    ciz("6) TAŞINABİLİRLİK — dosyaya derlendi, yeniden aç")
    boyut = os.path.getsize(yol)
    tekrar = mimar.open(yol, engines=[Echo()])
    print(f"    birlesik.mimar ({boyut} bayt) — provenans korundu:")
    for p in merge.verify_provenance(tekrar):
        print(f"      {p['tag']}: hash={p['hash']}")

ciz("7) ATTACH — bağımsız beyinler, bellekte birleşik görünüm (dosyalar değişmez)")
gorunum = merge.attach([kargo, iade])
print("    Attach görünümü:", gorunum.stats()["cache"], "niyet (A+B)")
print("    Kaynak dosyalar değişmedi (bağımsız kalır).")

ciz("BİTTİ")
