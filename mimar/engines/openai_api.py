"""
OpenAI-uyumlu API motoru.

OpenAI, OpenRouter, Together, Groq, yerel LM Studio/Ollama (OpenAI uyumlu uç) ve
diğerleri aynı /chat/completions şemasını konuşur; hepsi bu tek motordan geçer.
Sadece standart kütüphane (urllib) kullanır — ek bağımlılık yok.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Optional

from .base import Completion, Message


class OpenAIEngine:
    name = "openai"
    tier = "frontier"
    strength = 0.95          # maliyet modeli: güçlü model
    usd_per_call = 0.002     # kaba ortalama
    latency_ms = 1500.0

    def __init__(self, model: str = "gpt-4o-mini",
                 base_url: str = "https://api.openai.com/v1",
                 api_key: Optional[str] = None,
                 api_key_env: str = "OPENAI_API_KEY",
                 timeout: float = 60.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key) or "localhost" in self.base_url or "127.0.0.1" in self.base_url

    def complete(self, messages: list[Message], **opts) -> Completion:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role if m.role != "tool" else "system",
                          "content": m.content} for m in messages],
            "temperature": opts.get("temperature", 0.2),
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        return Completion(text=text, engine=f"{self.name}:{self.model}",
                          tokens=usage.get("total_tokens"), raw=body)
