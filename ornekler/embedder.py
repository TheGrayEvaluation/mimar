"""
Mimar — "Eş-anlam yapmazsak ne olur?" + takılabilir embedder.

Karar: Mimar elle eş-anlam SÖZLÜĞÜ tutmaz (kırılgan, her şeyi bozabilir). Bunun
yerine:
  1) Varsayılan hash embedder morfolojiyi yakalar; yakalamadığı derin eş-anlamda
     GÜVENLE modele düşer (yanlış cevap değil — sadece o parafraz öğrenilene
     kadar bir model çağrısı). Zamanla prototip öğrenme + feedback ile derlenir.
  2) İsteyen, veriden öğrenilmiş bir NÖRAL embedder'ı arayüze takar; eş-anlam
     doğru araçla (sözlükle değil) çözülür. Uzay tutarlılığı Mimar'ca korunur.

python ornekler/embedder.py
"""

import hashlib
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mimar
from mimar.engines import Echo
from mimar.selector import FunctionEmbedder


def ciz(b=""):
    print("\n" + "=" * 70)
    if b:
        print(b); print("=" * 70)


ciz("1) EŞ-ANLAM YAPMAZSAK — varsayılan hash, güvenli davranış")
b = mimar.open(None, engines=[Echo()])
b.declare(role="Destek")
b.teach("kargom ne zaman gelir", "Kargonuz 2-3 iş gününde gelir.")

print("  Morfolojik parafraz (hash yakalar):")
for q in ["kargom ne zaman gelecek", "kargo ne zaman gelir acaba"]:
    print(f"     güven={b.cache_confidence(q):<6} derlenir={b.cache_get(q) is not None}  «{q}»")
print("  Derin eş-anlam (hash yakalamaz → GÜVENLE modele düşer, yanlış cevap YOK):")
for q in ["paketim ne zaman ulaşır", "gönderim kaç günde elime geçer"]:
    print(f"     güven={b.cache_confidence(q):<6} derlenir={b.cache_get(q) is not None}  «{q}»  → model")

ciz("2) İSTEYEN NÖRAL EMBEDDER TAKAR — eş-anlam doğru araçla (sözlük değil)")
print("  (Gerçek nöral model API/yerel ister; burada mock 'nöral' embedder ile")
print("   sadece ARAYÜZ gösteriliyor — eş-anlam iddiası yok.)")


def mock_encode(text):
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return [x / 255.0 for x in h[:16]]


nb = mimar.open(None, engines=[Echo()],
                embedder=FunctionEmbedder(mock_encode, dim=16, id="mock-neural"))
nb.declare(role="Destek")
nb.teach("merhaba", "Hoş geldiniz!")
a = nb.run("merhaba")
print(f"  Özel embedder altında birebir reuse: [{a.source}] «merhaba» -> {a.text}")
print("  (Gerçek openai_embedder/yerel model aynı FunctionEmbedder kancasına takılır.)")

ciz("3) UZAY TUTARLILIĞI — yanlış embedder'la açış SESSİZCE BOZMAZ")
import tempfile
d = tempfile.mkdtemp()
yol = os.path.join(d, "beyin.mimar")
bb = mimar.open(yol, engines=[Echo()],
                embedder=FunctionEmbedder(mock_encode, 16, "mock-neural"))
bb.teach("x", "y"); bb.close()
try:
    mimar.open(yol, engines=[Echo()])           # varsayılan hash ≠ mock-neural
    print("  HATA: açılmamalıydı")
except ValueError as e:
    print(f"  ✓ net hata verdi (bozma yok): {str(e)[:70]}…")

ciz("ÖZET")
print("  • Eş-anlam sözlüğü YOK → kırılganlık yok, false-pozitif riski yok.")
print("  • Morfoloji hash'le gelir; derin eş-anlam güvenle modele düşer / nöral")
print("    embedder takılırsa çözülür. Uzay tutarlılığı korunur (yanlış açış = hata).")
