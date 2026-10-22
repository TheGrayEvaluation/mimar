"""
Mimar — Hafif Anlamsal Seçici (gömülü-dostu ML).

Bu, Mimar'ı "not defteri" olmaktan çıkaran parçadır. İki işi vardır ve HİÇBİRİ
metin üretmez; ikisi de KARAR verir:

  1) Anlamsal eşleştirme: yeni sorgu, daha önce derlenmiş bir niyetle aynı anlama
     mı geliyor? (Böylece birebir aynı cümle gerekmez — parafraz da tutar.)
  2) Güven: eşleşmeye güvenip ucuz yolu mu seçmeli, yoksa modele mi yükseltmeli?

Tasarım kısıtı: AĞIR BAĞIMLILIK YOK. Bir transformer değil; özellik-hashleme +
merkezcil (nearest-centroid) çevrimiçi öğrenen bir sınıflandırıcı. Saf aritmetik
olduğu için Rust/no_std ile mikrodenetleyiciye kadar taşınabilir. İstenirse gerçek
bir embedding modeli `Embedder` yerine takılabilir (ikinci katman).

Merkezcil öğrenme: her derlenmiş niyetin bir "prototip vektörü" vardır. Aynı niyete
ait yeni örnekler geldikçe prototip onlara doğru çekilir — yani eş anlamlılar
kanıt biriktikçe birbirine yakınsar. Küçük ama gerçek bir ML.
"""

from __future__ import annotations

import hashlib
import math
import re

DIM = 256                     # özellik uzayı boyutu (gömülü için küçük)
_WORD = re.compile(r"\w+", re.UNICODE)


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


def _char_ngrams(text: str, n: int) -> list[str]:
    t = "·" + " ".join(_words(text)) + "·"
    return [t[i:i + n] for i in range(len(t) - n + 1)]


def _features(text: str) -> list[str]:
    """Kelime + karakter n-gram karışımı. Karakter n-gram'lar Türkçe çekimleri
    yakalar (kargom~kargo, gelir~gelecek ortak parçalar)."""
    feats = [f"w:{w}" for w in _words(text)]
    for n in (3, 4):
        feats += [f"c{n}:{g}" for g in _char_ngrams(text, n)]
    return feats


def _h(feature: str) -> tuple[int, float]:
    d = hashlib.md5(feature.encode("utf-8")).digest()
    idx = int.from_bytes(d[:4], "little") % DIM
    sign = 1.0 if (d[4] & 1) else -1.0      # işaretli hashleme (sapmayı azaltır)
    return idx, sign


class Embedder:
    """Metni sabit boyutlu, L2-normalize bir vektöre çevirir (işaretli hashleme).
    Varsayılan, bağımlılıksız 1. katman: morfoloji/kelime örtüşmesini yakalar,
    eş-anlamı yakalamaz. Daha güçlü embedder'lar bu arayüze takılır (2. katman)."""

    dim = DIM
    id = "hash-256"

    def embed(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for f in _features(text):
            idx, sign = _h(f)
            v[idx] += sign
        return _normalize(v)


def _normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm > 0 else v


class FunctionEmbedder:
    """Takılabilir katman: herhangi bir `encode(text)->list[float]` çağrısını
    (yerel sentence-transformers, ONNX, uzak embedding API…) Mimar embedder'ına
    çevirir. Gerçek nöral model TAM OLARAK buraya takılır."""

    def __init__(self, encode, dim: int, id: str = "fn"):
        self._encode = encode
        self.dim = dim
        self.id = id

    def embed(self, text: str) -> list[float]:
        return _normalize(list(self._encode(text)))


def openai_embedder(model: str = "text-embedding-3-small",
                    base_url: str = "https://api.openai.com/v1",
                    api_key: str = "", dim: int = 1536):
    """OpenAI-uyumlu embedding uç noktası için hazır FunctionEmbedder (urllib).
    DÜRÜST NOT: burada ağ olmadan koşturulmadı; mimari kancanın hazır olduğunu
    gösterir — anahtar verilince gerçek nöral embedder devreye girer."""
    import json
    import os
    import urllib.request
    key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def encode(text: str):
        data = json.dumps({"model": model, "input": text}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/embeddings", data=data,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read())
        return body["data"][0]["embedding"]

    return FunctionEmbedder(encode, dim=dim, id=f"openai:{model}")


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))   # ikisi de normalize → nokta çarpımı


def mix_prototype(old: list[float], old_n: int, new_vec: list[float]) -> list[float]:
    """Çevrimiçi merkezcil güncelleme: prototipi yeni örneğe doğru çek, normalize et."""
    if not old:
        return new_vec[:]
    merged = [(o * old_n + nv) for o, nv in zip(old, new_vec)]
    norm = math.sqrt(sum(x * x for x in merged))
    if norm > 0:
        merged = [x / norm for x in merged]
    return merged
