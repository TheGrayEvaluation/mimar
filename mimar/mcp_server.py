"""
Mimar — MCP sunucusu (ajanlara montaj).

Mimar'ı Cursor/Antigravity/Windsurf/Hermes gibi MCP konuşan ajanlara takmanın yolu:
beyni bir Model Context Protocol sunucusu olarak açmak. Ajan, kodunu değiştirmeden
Mimar'ı araç + kaynak olarak mount eder; böylece tekrar eden/bilinen işleri büyük
modele hiç göndermeden Mimar'a devreder.

Taşıma: MCP stdio — satır-ayrımlı JSON-RPC 2.0 (bağımlılıksız). `handle()` saf bir
fonksiyondur (gerçek istemci olmadan test edilebilir); `serve_stdio()` gerçek
alt-süreç olarak koşar:  python -m mimar.mcp_server beyin.mimar

Sunduğu araçlar:
  mimar_ask      — beyne sor (derlenmiş/ucuz cevap, kaynak+maliyet döner)
  mimar_explain  — sorgu için planı göster (EXPLAIN)
  mimar_teach    — beyne bir girdi→çıktı öğret (ajan kendi tekrarını derler)
  mimar_feedback — son cevaba iyi/kötü geri bildirim (güvenli öğrenme)
Kaynak:
  mimar://stats  — beyin istatistiği (JSON)
"""

from __future__ import annotations

import json
import sys
from typing import Optional

from .brain import Brain
from .engines import Echo

PROTOCOL_VERSION = "2024-11-05"
SERVER_VERSION = "0.1.0"

TOOLS = [
    {"name": "mimar_ask",
     "description": "Beyne bir soru sor; bilineni ucuz/anında, bilinmeyeni modele "
                    "yönlendirir. Kaynak (cache/rule/model) ve maliyeti döner.",
     "inputSchema": {"type": "object",
                     "properties": {"query": {"type": "string"}},
                     "required": ["query"]}},
    {"name": "mimar_explain",
     "description": "Sorgu için seçilen planı (biliş operatörleri + maliyet) göster.",
     "inputSchema": {"type": "object",
                     "properties": {"query": {"type": "string"}},
                     "required": ["query"]}},
    {"name": "mimar_teach",
     "description": "Beyne bir girdi→çıktı öğret (tekrar eden işi kalıcı derle).",
     "inputSchema": {"type": "object",
                     "properties": {"prompt": {"type": "string"},
                                    "answer": {"type": "string"}},
                     "required": ["prompt", "answer"]}},
    {"name": "mimar_feedback",
     "description": "Bir sorgunun cevabına iyi/kötü geri bildirim (güvenli öğrenme; "
                    "kötü → niyet engellenir ve sökülür).",
     "inputSchema": {"type": "object",
                     "properties": {"query": {"type": "string"},
                                    "good": {"type": "boolean"}},
                     "required": ["query", "good"]}},
]

RESOURCES = [
    {"uri": "mimar://stats", "name": "Beyin istatistiği",
     "description": "Beynin kural/niyet/çalışma istatistikleri",
     "mimeType": "application/json"},
]


class MimarMCP:
    """Bir beyni MCP üzerinden sunan sunucu çekirdeği."""

    def __init__(self, brain: Brain):
        self.brain = brain

    # ---- JSON-RPC yönlendirme -------------------------------------------
    def handle(self, msg: dict) -> Optional[dict]:
        method = msg.get("method")
        mid = msg.get("id")
        if method is None:          # yanıt/bilinmeyen → yok say
            return None
        try:
            if method == "initialize":
                result = self._initialize()
            elif method in ("notifications/initialized", "initialized"):
                return None          # bildirim: yanıt yok
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                result = self._call(msg.get("params", {}))
            elif method == "resources/list":
                result = {"resources": RESOURCES}
            elif method == "resources/read":
                result = self._read(msg.get("params", {}))
            else:
                return self._err(mid, -32601, f"method not found: {method}")
        except Exception as e:       # JSON-RPC iç hata
            return self._err(mid, -32603, str(e))
        if mid is None:
            return None
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    # ---- yöntemler ------------------------------------------------------
    def _initialize(self) -> dict:
        return {"protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}, "resources": {}},
                "serverInfo": {"name": "mimar", "version": SERVER_VERSION}}

    def _call(self, params: dict) -> dict:
        name = params.get("name")
        args = params.get("arguments", {}) or {}
        if name == "mimar_ask":
            a = self.brain.run(args["query"])
            txt = (f"{a.text}\n\n[mimar] kaynak={a.source} "
                   f"maliyet={a.cost.ms:.2f}ms engellendi={a.blocked}")
            return self._text(txt)
        if name == "mimar_explain":
            return self._text(str(self.brain.explain(args["query"])))
        if name == "mimar_teach":
            self.brain.teach(args["prompt"], args["answer"])
            return self._text("öğretildi (kalıcı niyete derlendi)")
        if name == "mimar_feedback":
            return self._text(self.brain.feedback(args["query"],
                                                  bool(args["good"])))
        return self._text(f"bilinmeyen araç: {name}", is_error=True)

    def _read(self, params: dict) -> dict:
        uri = params.get("uri")
        if uri == "mimar://stats":
            return {"contents": [{"uri": uri, "mimeType": "application/json",
                                  "text": json.dumps(self.brain.stats(),
                                                      ensure_ascii=False)}]}
        raise ValueError(f"bilinmeyen kaynak: {uri}")

    # ---- yardımcılar ----------------------------------------------------
    @staticmethod
    def _text(text: str, is_error: bool = False) -> dict:
        return {"content": [{"type": "text", "text": text}], "isError": is_error}

    @staticmethod
    def _err(mid, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": mid,
                "error": {"code": code, "message": message}}


def serve_stdio(brain: Brain) -> None:
    """Gerçek MCP stdio döngüsü: satır-ayrımlı JSON-RPC."""
    srv = MimarMCP(brain)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = srv.handle(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass
    path = sys.argv[1] if len(sys.argv) > 1 else None
    serve_stdio(Brain(path, [Echo()]))
