"""
Mimar — Konsolidasyon kanıtı (şişmeyen beyin).

Ağır kullanımı taklit eder: 5 kalıcı (öğretilmiş) bilgi + 300 geçici/farklı sorgu.
Dosya şişer. Sonra konsolide eder ve ÖLÇER: dosya küçülür mü, DAVRANIŞ (öğretilen
5 bilgi) korunur mu?

python ornekler/konsolidasyon.py
"""

import os
import random
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mimar
from mimar.engines import Echo


def ciz(b=""):
    print("\n" + "=" * 70)
    if b:
        print(b); print("=" * 70)


random.seed(7)
AD = ["sipariş", "fatura", "kargo", "ürün", "hesap", "kampanya", "stok", "bayi",
      "kupon", "teslimat", "garanti", "servis", "iade", "ödeme", "adres", "paket",
      "indirim", "üyelik", "puan", "şube", "destek", "sistem", "uygulama", "site"]
FIIL = ["nasıl", "nedir", "ne zaman", "nereden", "kaç", "neden", "hangi", "kim"]
EK = ["olur", "yapılır", "görünür", "değişir", "bulunur", "gelir", "açılır", "kapanır"]


def rastgele_sorgu():
    return f"{random.choice(AD)} {random.choice(AD)} {random.choice(FIIL)} {random.choice(EK)}"


d = tempfile.mkdtemp()
yol = os.path.join(d, "yogun.mimar")
beyin = mimar.open(yol, engines=[Echo()])
beyin.declare(role="Yoğun kullanılan destek beyni.")

ciz("1) KALICI BİLGİ — 5 öğretilmiş davranış (korunmalı)")
kanon = {
    "kargom ne zaman gelir": "2-3 iş günü.",
    "ürünü nasıl iade ederim": "14 gün ücretsiz iade.",
    "şifremi nasıl sıfırlarım": "E-postadan sıfırlama bağlantısı.",
    "faturamı nereden görürüm": "Hesabım > Faturalar.",
    "kampanyalar nedir": "Hafta sonu %20 indirim.",
}
for q, a in kanon.items():
    beyin.teach(q, a)
print(f"   {len(kanon)} kalıcı bilgi öğretildi")

ciz("2) AĞIR KULLANIM — 300 farklı geçici sorgu")
for _ in range(300):
    beyin.run(rastgele_sorgu())
beyin.save()
s1 = os.path.getsize(yol)
n1 = beyin.stats()["cache"]
m1 = beyin.stats()["memory"]
print(f"   Önce:  {n1} niyet, {m1} hafıza kaydı, DOSYA {s1:,} bayt")

ciz("3) KONSOLİDE ET — benzerleri birleştir, uzun-kuyruğu buda, sınırla")
rapor = beyin.consolidate(max_intents=20, max_memory=30)
s2 = os.path.getsize(yol)
print("  ", rapor)
print(f"   Sonra: {rapor['niyet_sonra']} niyet, {rapor['hafıza_sonra']} hafıza, "
      f"DOSYA {s2:,} bayt")
print(f"\n   >>> DOSYA {s1/s2:.1f}x küçüldü "
      f"({s1:,} → {s2:,} bayt), korunan davranış: {rapor['korunan']} niyet")

ciz("4) DOĞRULUK KORUNDU MU? — 5 kalıcı bilgiyi tekrar sor")
dogru = 0
for q, a in kanon.items():
    cevap = beyin.run(q)
    ok = a in cevap.text
    dogru += 1 if ok else 0
    print(f"   {'✓' if ok else '✗'} [{cevap.source}] «{q}» -> {cevap.text}")
print(f"\n   Kalıcı bilgi doğruluğu: {dogru}/{len(kanon)} "
      f"({'KORUNDU' if dogru == len(kanon) else 'BOZULDU'})")

ciz("ÖZET")
print(f"   {n1} niyet → {rapor['niyet_sonra']} niyet,  dosya {s1/s2:.1f}x küçük,  "
      f"davranış {dogru}/{len(kanon)} korundu.")
print("   Beyin kullanıldıkça AKILLANIR ama ŞİŞMEZ — deneyim satır değil, öz olarak yaşar.")
