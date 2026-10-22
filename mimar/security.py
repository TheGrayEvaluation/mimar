"""
Mimar — yerleşik güvenlik.

Güvenlik kütüphanenin isteğe bağlı bir eklentisi değil, planlayıcının içinden
geçtiği bir kapıdır. İlkeler:

  1) En az yetki: bir araç ancak hem BİLDİRİLMİŞ hem de açıkça BAĞLANMIŞsa çağrılır.
     Dosyadaki bildirim koddan ayrıdır; dosya tek başına kod çalıştıramaz.
  2) Bütçe: her çalışma için model çağrısı / araç çağrısı üst sınırı. Sonsuz döngü
     ve maliyet patlaması engellenir.
  3) Guard kuralları: üretilen çıktı, dosyadaki yasak desenlere karşı denetlenir;
     eşleşirse yanıt bloklanır.
  4) PII maskeleme: modele gitmeden önce hassas desenler maskelenebilir.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# Kaba ama kullanışlı PII desenleri (genişletilebilir)
_PII = {
    "email": re.compile(r"[\w.\-]+@[\w\-]+\.[\w.\-]+"),
    "tc_kimlik": re.compile(r"\b\d{11}\b"),
    "kart": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "iban": re.compile(r"\bTR\d{2}[0-9 ]{20,30}\b", re.IGNORECASE),
}


class SecurityError(Exception):
    pass


@dataclass
class Policy:
    """Bir çalışmanın güvenlik + maliyet sözleşmesi.

    Optimizasyoncu planı bu sözleşmeye göre seçer: risk düşükse en ucuz yolu,
    risk yüksekse en güçlü motor + zorunlu doğrulamayı tercih eder.
    """
    max_model_calls: int = 4
    max_tool_calls: int = 8
    allow_tools: Optional[set[str]] = None   # None = bağlı olan her araç serbest
    redact_pii: bool = False
    dev_mode: bool = False                    # programın kendini geliştirmesi
    # --- maliyet modeli kısıtları ---
    max_cost_usd: float = 1.0                 # tek çalışma için para üst sınırı
    max_latency_ms: float = 30_000.0          # kabul edilebilir gecikme
    risk: str = "normal"                      # low | normal | high (yüksek=doğrula+güçlü motor)
    min_strength: float = 0.0                 # gereken asgari model gücü (0..1)

    def tool_allowed(self, name: str, bound: set[str]) -> bool:
        if name not in bound:
            return False
        if self.allow_tools is None:
            return True
        return name in self.allow_tools


@dataclass
class Budget:
    """Çalışma sırasında tüketilen kaynak sayacı."""
    policy: Policy
    model_calls: int = 0
    tool_calls: int = 0

    def spend_model(self) -> None:
        self.model_calls += 1
        if self.model_calls > self.policy.max_model_calls:
            raise SecurityError(
                f"Model çağrısı bütçesi aşıldı (> {self.policy.max_model_calls}).")

    def spend_tool(self) -> None:
        self.tool_calls += 1
        if self.tool_calls > self.policy.max_tool_calls:
            raise SecurityError(
                f"Araç çağrısı bütçesi aşıldı (> {self.policy.max_tool_calls}).")


def redact(text: str) -> str:
    """Hassas desenleri maskele."""
    out = text
    for label, rx in _PII.items():
        out = rx.sub(f"[{label}]", out)
    return out


def check_guards(text: str, forbid_patterns: list[str]) -> Optional[str]:
    """Çıktı bir guard desenini ihlal ediyor mu? İhlal eden deseni döndür."""
    for pat in forbid_patterns:
        if not pat:
            continue
        try:
            if re.search(pat, text, re.IGNORECASE):
                return pat
        except re.error:
            # Desen geçerli regex değilse düz metin olarak ara
            if pat.lower() in text.lower():
                return pat
    return None
