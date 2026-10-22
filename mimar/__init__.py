"""
Mimar — Yapay zekânın SQLite'ı.

Her programa gömülü, sıfır kurulumla çalışan, programa yapay zekâ kazandıran
açık kaynak bir çekirdek. İki kalbi vardır:

  1) Planlayıcı/Optimizasyoncu: Geliştirici NİYETİ bildirir (rol, bilgi, kurallar,
     araçlar, risk/bütçe). Mimar bunu bir biliş baytkoduna derler ve bir maliyet
     modeline göre en ucuz ama riski karşılayan planı seçer — tıpkı SQL'in sorgu
     planlayıcısı gibi.

  2) Tek dosyalık beyin (.mimar): Programın bütün yapay zekâsı — rol, araçlar,
     bilgi indeksi, hafıza, LoRA, önbellek, istatistik — tek taşınabilir dosyada
     yaşar. Kullanıldıkça kendini derler: daha akıllı ve daha ucuz olur.

Model bağımsızdır: yerel model de API de takılabilen motorlardan biridir.

Hızlı başlangıç:
    import mimar
    from mimar.engines import Echo

    beyin = mimar.open("destek.mimar", engines=Echo())
    beyin.declare(
        role="Acme için müşteri destek asistanı.",
        guards=["asla iade sözü verme"],
        knowledge=["Kargo 2-3 iş gününde teslim edilir."],
    )
    print(beyin.explain("kargom nerede"))   # planı göster (EXPLAIN gibi)
    print(beyin.ask("kargom nerede"))        # çalıştır
"""

from __future__ import annotations

from typing import Optional, Union

from . import engines
from .brain import Brain
from .engines import Echo
from .security import Policy
from .vm import Answer, Op, Plan

__version__ = "0.1.0"
__all__ = ["open", "Brain", "Policy", "Plan", "Answer", "Op", "engines"]


def open(path: Optional[str] = None, engines=None,
         policy: Optional[Policy] = None, autosave: bool = True,
         embedder=None) -> Brain:
    """Bir .mimar beynini aç (yoksa oluştur). SQLite'ın sqlite3.connect'i gibi.

    path=None ise bellekte geçici bir beyin açılır (diske yazılmaz).
    embedder=None ise varsayılan hash embedder (bağımlılıksız). Güçlü/nöral/lexicon
    embedder opt-in olarak buradan verilir; uzay tutarlılığı Brain tarafından korunur.
    """
    eng = engines if engines is not None else [Echo()]
    return Brain(path, eng, policy=policy, autosave=autosave, embedder=embedder)
