"""
Mimar — Anlamsal indeks (LSH) ölçümü.

Niyet sayısı N büyüdükçe:
  - Kaba (lineer) arama O(N): yavaşlar.
  - LSH indeksi: sorgu süresi ~sabit, recall yüksek.

Ölçer: her N için kaba vs indeks sorgu süresi (ms/sorgu) ve recall@1
(indeksin bulduğu en-yakın == kabanın bulduğu en-yakın mı). Perturbe edilmiş
sorgularla gerçek recall sınanır (birebir değil).

python ornekler/indeks.py
"""

import math
import os
import random
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mimar.index import LSHIndex
from mimar.selector import Embedder, cosine

emb = Embedder()
rng = random.Random(42)

# Ayrık (ortak kök TAŞIMAYAN) rastgele kelimeler — gerçek sözcük çeşitliliğini
# taklit eder (yoksa ortak kök tüm vektörleri yapay olarak benzeştirir).
_HARF = "abcdefghijklmnoprstuvyz"
POOL = list({"".join(rng.choice(_HARF) for _ in range(rng.randint(4, 8)))
             for _ in range(600)})


def rastgele_metin():
    return " ".join(rng.sample(POOL, 5))


def paraphrase_metin(metin):
    """Gerçekçi parafraz: 5 kelimeden birini değiştir (4/5 ortak → yüksek benzerlik)."""
    kel = metin.split()
    kel[rng.randrange(len(kel))] = rng.choice(POOL)
    return " ".join(kel)


def brute_nearest(qv, vecs, keys):
    best_k, best_s = None, -1.0
    for k, v in zip(keys, vecs):
        s = cosine(qv, v)
        if s > best_s:
            best_k, best_s = k, s
    return best_k


def ciz(b=""):
    print("\n" + "=" * 70)
    if b:
        print(b); print("=" * 70)


ciz("LSH İNDEKS ÖLÇÜMÜ (saf Python, bağımlılıksız)")
print(f"{'N':>6} | {'kaba ms/sorgu':>14} | {'indeks ms/sorgu':>16} | "
      f"{'hızlanma':>9} | {'recall@1':>9} | {'ort aday':>9}")
print("-" * 78)

for N in [128, 512, 2000, 5000]:
    metinler = [rastgele_metin() for _ in range(N)]
    keys = [f"k{i}" for i in range(N)]
    vecs = [emb.embed(m) for m in metinler]
    idx = LSHIndex(emb.dim)
    for k, v in zip(keys, vecs):
        idx.add(k, v)

    # Sorgular: mevcut vektörlerin perturbe hali (gerçek en-yakın = kaynak)
    Q = 40
    sorgu_idx = [rng.randrange(N) for _ in range(Q)]
    sorgular = [emb.embed(paraphrase_metin(metinler[i])) for i in sorgu_idx]

    t = time.perf_counter()
    brute_res = [brute_nearest(qv, vecs, keys) for qv in sorgular]
    brute_ms = (time.perf_counter() - t) * 1000 / Q

    t = time.perf_counter()
    idx_res, aday_say = [], 0
    for qv in sorgular:
        cand = idx.candidates(qv)
        aday_say += len(cand)
        best_k, best_s = None, -1.0
        for k in cand:
            s = cosine(qv, vecs[keys.index(k)])
            if s > best_s:
                best_k, best_s = k, s
        idx_res.append(best_k)
    idx_ms = (time.perf_counter() - t) * 1000 / Q

    recall = sum(1 for a, b in zip(brute_res, idx_res) if a == b) / Q
    speed = brute_ms / idx_ms if idx_ms else float("inf")
    # Ortalama sorgu-kaynak benzerliği (recall'ı bağlama oturtmak için)
    avg_sim = sum(cosine(sorgular[j], vecs[sorgu_idx[j]]) for j in range(Q)) / Q
    print(f"{N:>6} | {brute_ms:>14.3f} | {idx_ms:>16.3f} | {speed:>8.1f}x | "
          f"{recall*100:>7.0f}% | {aday_say/Q:>9.0f}  (ort benzerlik {avg_sim:.2f})")

ciz("SONUÇ")
print("  Kaba arama N ile lineer büyür; indeks sorgu süresi ~sabit kalır.")
print("  (Not: keys.index() demo kolaylığı; gerçekte anahtar→vektör haritası O(1).)")
print("  Üretimde: birebir/öğretilmiş eşleşme ayrıca anahtarla kesin bulunur →")
print("  LSH kaçırsa bile yanlış cevap değil, en kötü modele düşülür.")
