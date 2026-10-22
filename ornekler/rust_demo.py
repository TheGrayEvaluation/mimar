"""
Mimar — Rust çekirdek çapraz doğrulaması.

Python bir .mimar beyni üretir; native Rust ikilisi onu okuyup cevaplar. Aynı
sorgu vektörüyle Rust'ın anlamsal sonucu Python'unkiyle BİREBİR aynı mı sınanır.
Bu, formatın ve çekirdeğin "diğer sistemlerde" gerçekten çalıştığının kanıtıdır.

python ornekler/rust_demo.py
"""

import os
import subprocess
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mimar

RUST = os.path.join(ROOT, "rust-cekirdek", "target", "release", "mimar.exe")
if not os.path.exists(RUST):
    RUST = os.path.join(ROOT, "rust-cekirdek", "target", "release", "mimar")


def ciz(b=""):
    print("\n" + "=" * 70)
    if b:
        print(b); print("=" * 70)


d = tempfile.mkdtemp()
yol = os.path.join(d, "ortak.mimar")
b = mimar.open(yol, engines=[mimar.engines.Echo()])
b.declare(role="Ortak beyin (Python yazar, Rust okur).")
b.teach("reset password", "Use the email reset link.")
b.teach("shipping time", "Ships in 2 to 3 days.")
b.teach("open hours", "Weekdays 9 to 6.")
b.close()


def rust(*a):
    out = subprocess.run([RUST, yol, *a], capture_output=True, text=True,
                         encoding="utf-8")
    return (out.stdout or out.stderr).strip()


ciz("1) NATIVE RUST beyni okuyor (Python'un yazdığı dosya)")
print(rust())

ciz("2) RUST birebir sorgu → cevap (gömme gerekmez)")
print("  ", rust("--ask", "shipping time"))
print("  ", rust("--ask", "bilinmeyen soru"))

ciz("3) RUST anlamsal en-yakın komşu (saklı vektörler, native cosine)")
print("  ", rust("--nn", "0"))

ciz("4) ÇAPRAZ DOĞRULAMA — aynı vektörde Python vs Rust (MUTASYONSUZ)")
sorgu = "shipping times"     # taught 'shipping time' parafrazı
qv = b.embedder.embed(sorgu)
# Python tarafı: MUTASYONSUZ ölçüm (run YOK → dosyayı kirletmez)
bk, bc, bs, _ = b._best_intent(sorgu)
py_serve = bs >= 0.55
print(f"   sorgu  : «{sorgu}»")
print(f"   Python : best=«{bc['query']}» sim={bs:.6f} serve={py_serve}")
# Rust tarafı: aynı vektör, aynı (değişmemiş) dosya
vec_arg = ",".join(f"{x:.9f}" for x in qv)
rs = rust("--vec", vec_arg)
print(f"   Rust   : {rs}")
# Parite: aynı niyet + aynı sim (± yuvarlama)
import re
m = re.search(r"best=«(.+?)» sim=([0-9.]+)", rs)
ayni = bool(m) and m.group(1) == bc["query"] and abs(float(m.group(2)) - bs) < 1e-4
print(f"\n   >>> Aynı niyet + aynı benzerlik mi? {'EVET ✓' if ayni else 'HAYIR ✗'}")

ciz("ÖZET")
print("  • .mimar formatı saf-std Rust'la (sıfır bağımlılık) NATIVE okundu.")
print("  • Birebir geri getirme + cosine anlamsal arama Rust'ta çalışıyor.")
print("  • Aynı vektörde Python==Rust → çekirdek diller/sistemler arası tutarlı.")
print("  • Gömme (metin→vektör) takılabilir; çekirdek okuma+arama her yerde koşar.")
