"""
Yerel model motoru — llama.cpp / LM Studio / Ollama.

Pratikte iki yol vardır:
  1) Yerel sunucu OpenAI-uyumlu bir uç açar (LM Studio, Ollama `/v1`, llama.cpp
     server). Bu durumda OpenAIEngine'i localhost'a yöneltmek yeterlidir; bu sınıf
     onu ergonomik biçimde sarar.
  2) Süreç-içi llama.cpp bağlaması (llama-cpp-python) kuruluysa doğrudan onu kullanır.

Hiçbiri yoksa available() False döner ve planlayıcı başka motora/kurala düşer.
"""

from __future__ import annotations

from typing import Optional

from .base import Completion, Message
from .openai_api import OpenAIEngine


class LocalEngine:
    name = "local"
    tier = "local"
    strength = 0.6           # maliyet modeli: orta güç, ucuz
    usd_per_call = 0.0       # kendi donanımın
    latency_ms = 400.0

    def __init__(self, model: str = "local-model",
                 base_url: str = "http://127.0.0.1:1234/v1",
                 gguf_path: Optional[str] = None):
        self.model = model
        self.gguf_path = gguf_path
        self._llama = None
        self._http = OpenAIEngine(model=model, base_url=base_url,
                                  api_key="not-needed")
        if gguf_path:
            try:
                from llama_cpp import Llama  # type: ignore
                self._llama = Llama(model_path=gguf_path, n_ctx=4096, verbose=False)
            except Exception:
                self._llama = None

    def available(self) -> bool:
        if self._llama is not None:
            return True
        try:
            import socket
            from urllib.parse import urlparse
            u = urlparse(self._http.base_url)
            with socket.create_connection((u.hostname, u.port or 80), timeout=0.5):
                return True
        except Exception:
            return False

    def complete(self, messages: list[Message], **opts) -> Completion:
        if self._llama is not None:
            chat = [{"role": m.role if m.role != "tool" else "system",
                     "content": m.content} for m in messages]
            out = self._llama.create_chat_completion(
                messages=chat, temperature=opts.get("temperature", 0.2))
            text = out["choices"][0]["message"]["content"]
            return Completion(text=text, engine=f"{self.name}:gguf", raw=out)
        return self._http.complete(messages, **opts)
