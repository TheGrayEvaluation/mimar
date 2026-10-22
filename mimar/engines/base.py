"""
Mimar motor arayüzü — model bağımsızlığının kalbi.

Bir "motor" tek bir sözleşmeyi yerine getirir: mesajları al, metin döndür.
Planlayıcı hangi motorun takılı olduğunu bilmez; yerel model de, uzak API de,
hatta kural tabanlı sahte bir motor da aynı arayüzü konuşur.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class Message:
    role: str      # system | user | assistant | tool
    content: str


@dataclass
class Completion:
    text: str
    engine: str
    tokens: Optional[int] = None
    raw: Optional[dict] = None


class Engine(Protocol):
    """Her motor bu sözleşmeyi uygular."""

    name: str

    def complete(self, messages: list[Message], **opts) -> Completion:
        ...

    def available(self) -> bool:
        """Motor şu an kullanılabilir mi? (ör. API anahtarı/sunucu var mı)"""
        ...
