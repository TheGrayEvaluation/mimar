"""
Mimar — Davranış damıtma kanıtı.

Senaryo: Bir "usta" modelin kendine özgü bir konuşma tarzı var. Onunla birkaç
sohbet yapıp (yakalama) bir .mimar dosyasına damıtıyoruz. Sonra bu dosyayı, o
tarzı HİÇ bilmeyen BAŞKA motorlara takıp davranışın yeniden üretilip üretilmediğini
EVAL ile ölçüyoruz.

Beklenen dürüst sonuç:
  - Beyin yokken yabancı motor: ~%0 (tarzı bilmiyor).
  - Beyin takılıyken: yakalanan niyetlerde ~%100 (davranış dosyadan geliyor).
  - İkinci farklı motorda da ~%100 (taşınabilir: davranış motorda değil DOSYADA).
  - Parafrazlarda: kısmi (anlamsal genelleme — dürüst sınır).

python ornekler/davranis.py
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mimar
from mimar import davranis
from mimar.engines import Completion, Echo, Message
from mimar.selector import Embedder


class UstaEngine(Echo):
    """Hedef davranış: taklidi zor, kendine özgü bir persona."""
    name = "usta"
    tier = "usta"
    strength = 0.9

    def complete(self, messages, **o):
        u = next((m.content for m in reversed(messages) if m.role == "user"), "")
        txt = f"USTA DİYOR: «{u.strip()}» — tabii dostum, hemen hallederiz. 🦊"
        return Completion(text=txt, engine=self.name)


def ciz(b=""):
    print("\n" + "=" * 70)
    if b:
        print(b); print("=" * 70)


emb = Embedder()
usta = UstaEngine()

prompts = [
    "kargom ne zaman gelir",
    "ürünü nasıl iade ederim",
    "şifremi unuttum ne yapmalıyım",
    "faturamı nereden görebilirim",
    "mağaza saatleri nedir",
    "siparişimi iptal etmek istiyorum",
]

ciz("1) YAKALA — usta modelle 6 sohbet (eğitim+test verisi)")
pairs = davranis.capture(usta, prompts)
for p, o in pairs[:2]:
    print(f"   «{p}» -> {o}")
print("   ... (6 çift)")

ciz("2) DAMIT — sohbetleri tek .mimar dosyasına al")
beyin = mimar.open(None, engines=[Echo()])          # NÖTR motor takılı (usta değil)
ozet = davranis.ingest(beyin, pairs, persona="Usta gibi sıcak ve kısa konuş.")
print("  ", ozet)

seen = [(p, o) for p, o in pairs]

ciz("3) ÖLÇ — davranış dosyadan yeniden üretiliyor mu?")
raw = davranis.raw_fidelity(Echo(), beyin.system_prompt(), seen, emb)
print(f"  Beyin YOK (nötr motor)     : sadakat {raw['oran']*100:.0f}%  "
      f"({raw['uyan']}/{raw['toplam']})  <-- tarzı bilmiyor")
wb = davranis.fidelity(beyin, Echo(), seen)
print(f"  Beyin VAR (nötr motor)     : sadakat {wb['oran']*100:.0f}%  "
      f"({wb['uyan']}/{wb['toplam']})  <-- davranış DOSYADAN geldi")

ciz("4) TAŞINABİLİRLİK — AYNI dosya, BAŞKA bir motor")
wb2 = davranis.fidelity(beyin, Echo(prefix="FARKLI-MOTOR"), seen)
print(f"  Beyin VAR (2. farklı motor): sadakat {wb2['oran']*100:.0f}%  "
      f"({wb2['uyan']}/{wb2['toplam']})  <-- davranış motorda değil, dosyada")

ciz("5) GENELLEME — parafrazlar (anlamsal, dürüst sınır)")
out = {p: o for p, o in pairs}
para = [
    ("kargo ne zaman gelecek", out[prompts[0]]),
    ("şifremi unuttum ne yapabilirim", out[prompts[2]]),
    ("siparişimi iptal etmek istiyorum lütfen", out[prompts[5]]),
]
g = davranis.fidelity(beyin, Echo(), para)
print(f"  Parafraz sadakati          : {g['oran']*100:.0f}%  "
      f"({g['uyan']}/{g['toplam']})")
for d in g["detay"]:
    print(f"     {'✓' if d['uydu'] else '·'} benz={d['benzerlik']:<5} «{d['girdi']}»")

ciz("6) EN YÜKSEK SADAKAT KATMANI — LoRA yeniden-damıtma (dürüst iskelet)")
print("  ", davranis.redistill_lora(beyin, usta))
print("   Beyin dosyasındaki adapter yuvaları:", list(beyin.blobs.keys()))

ciz("ÖZET")
print(f"  Beyin yok {raw['oran']*100:.0f}%  →  beyin var {wb['oran']*100:.0f}%  "
      f"→  başka motor {wb2['oran']*100:.0f}%  →  parafraz {g['oran']*100:.0f}%")
print("  Davranış, modelden bağımsız biçimde TEK DOSYAYA taşındı ve ölçülerek")
print("  doğrulandı. (Gerçek LLM sadakati gerçek modelle ölçülür; burada mekanizma")
print("   ve ölçüm iskeleti kanıtlandı.)")
