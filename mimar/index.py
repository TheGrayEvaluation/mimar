"""
Mimar — Anlamsal İndeks (LSH, rastgele hiper-düzlem).

Anlamsal arama varsayılan olarak lineer tarama yapar (her niyetle cosine). Bu,
niyet sayısı N büyüdükçe O(N) olur — milyon konuşmada yavaş. LSH (Locality-
Sensitive Hashing), kosinüs için klasik yaklaşık-en-yakın-komşu yöntemidir:

  - Sabit (tohumlu, yeniden üretilebilir) rastgele hiper-düzlemler. Her vektör,
    her düzlemin hangi tarafında olduğuna göre bir BİT imzası alır.
  - L tane tablo, her biri K bitlik alt-imzayı kova anahtarı yapar. Benzer
    vektörler aynı kovaya düşer. Sorgu sadece eşleşen kovalardaki ADAYLARA bakar;
    tam cosine yalnız onlar üzerinde hesaplanır.

Böylece sorgu maliyeti N'den (neredeyse) bağımsız, L·K·dim sabitine iner.
Bağımlılıksız, saf aritmetik → Rust/gömülüye taşınabilir.

GÜVENLİK: LSH YAKLAŞIKTIR, gerçek en-yakını kaçırabilir. Bu yüzden Mimar'da
birebir/öğretilmiş eşleşmeler ayrıca ANAHTARLA kesin bulunur (brain._best_intent);
LSH sadece parafraz adaylarını getirir. Kaçırırsa en kötü ihtimalle modele düşülür
(yanlış cevap verilmez) — abstention güvenliği korunur.
"""

from __future__ import annotations

import random


class LSHIndex:
    def __init__(self, dim: int, n_tables: int = 16, bits: int = 10,
                 seed: int = 1_234_567):
        self.dim = dim
        self.L = n_tables
        self.K = bits
        rng = random.Random(seed)
        # planes[t][i] = i. hiper-düzlem (dim boyut), Gauss dağılımlı
        self.planes = [[[rng.gauss(0, 1) for _ in range(dim)]
                        for _ in range(bits)] for _ in range(n_tables)]
        self.tables: list[dict] = [{} for _ in range(n_tables)]
        self.size = 0

    def _sig(self, vec: list, t: int) -> int:
        planes = self.planes[t]
        s = 0
        for i in range(self.K):
            h = planes[i]
            dot = 0.0
            for a, b in zip(vec, h):
                dot += a * b
            if dot >= 0.0:
                s |= (1 << i)
        return s

    def add(self, key: str, vec: list) -> None:
        for t in range(self.L):
            self.tables[t].setdefault(self._sig(vec, t), []).append(key)
        self.size += 1

    def candidates(self, vec: list) -> set:
        c = set()
        for t in range(self.L):
            c.update(self.tables[t].get(self._sig(vec, t), ()))
        return c
