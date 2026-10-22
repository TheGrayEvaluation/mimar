"""
Mimar — Beyin imzalama (dağıtım güveni) kanıtı.

python ornekler/imza.py
"""

import os
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


d = tempfile.mkdtemp()
yol = os.path.join(d, "dagitim.mimar")
b = mimar.open(yol, engines=[Echo()])
b.declare(role="Dağıtılacak uzman beyin.")
b.teach("x nasıl yapılır", "Şöyle yapılır: ...")

ciz("1) İMZALA — dağıtımdan önce (kimlik + kurcalama koruması)")
s = b.sign(key="gizli-anahtar-123", author="ufuk@mimar")
print(f"   imza: alg={s['alg']} author={s['author']}")
print(f"   content_hash={s['content_hash'][:24]}…")

ciz("2) DOĞRULA — doğru anahtar")
print("  ", b.verify("gizli-anahtar-123"))

ciz("3) DOĞRULA — YANLIŞ anahtar (sahte kaynak)")
print("  ", b.verify("yanlis-anahtar"))

ciz("4) KURCALAMA — biri içeriği değiştirirse yakalanır")
b.data["cache"][list(b.data["cache"])[0]]["answer"] = "KÖTÜ NIYETLİ CEVAP"
print("   (bir derlenmiş cevap gizlice değiştirildi)")
print("  ", b.verify("gizli-anahtar-123"))

ciz("5) ROUND-TRIP — imzala, diske yaz, yeniden aç, doğrula")
b2 = mimar.open(yol, engines=[Echo()])          # değişiklik kaydedilmişti; yeniden imzala
b2.declare()
b2.teach("temiz soru", "temiz cevap")
b2.sign(key="k2", author="dağıtıcı")
b2.close()
b3 = mimar.open(yol, engines=[Echo()])
print("   yeniden açıldı →", b3.verify("k2"))

ciz("ÖZET")
print("  • Dağıtılan beyin imzalı: kurcalama yakalanır, kaynak doğrulanır.")
print("  • HMAC-SHA256 (bağımlılıksız); asimetrik Ed25519/PKI üretim yükseltmesi.")
print("  • İmza yalnız bildirimi korur; araç KODU zaten dosyada taşınmaz.")
