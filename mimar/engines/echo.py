"""
Echo motoru — sıfır bağımlılık, çevrimdışı.

Gerçek bir model değildir; planlayıcıyı, kuralları, bilgi aramasını ve güvenlik
katmanını hiçbir API anahtarı/indirme olmadan uçtan uca test etmek içindir.
Kendisine verilen bağlamı (sistem istemi + bulunan bilgi) özetleyerek deterministik,
incelenebilir bir yanıt üretir. Testlerin ağ olmadan geçmesini sağlar.
"""

from __future__ import annotations

from .base import Completion, Message


class Echo:
    name = "echo"
    tier = "mock"
    strength = 0.2          # maliyet modeli: güç (0..1)
    usd_per_call = 0.0      # bedava
    latency_ms = 1.0        # anında

    def __init__(self, prefix: str = "[echo]"):
        self.prefix = prefix

    def available(self) -> bool:
        return True

    def complete(self, messages: list[Message], **opts) -> Completion:
        system = next((m.content for m in messages if m.role == "system"), "")
        user = next((m.content for m in reversed(messages)
                     if m.role == "user"), "")
        ctx = [m.content for m in messages if m.role == "tool"]
        parts = [f"{self.prefix} soru: {user.strip()}"]
        if ctx:
            parts.append("dayanak: " + " | ".join(c.strip() for c in ctx))
        if system:
            first = system.strip().splitlines()[0]
            parts.append(f"rol: {first}")
        text = "\n".join(parts)
        return Completion(text=text, engine=self.name, tokens=len(text.split()))
