"""
Mimar — MCP montaj kanıtı.

Mimar sunucusunu GERÇEK bir alt-süreç olarak başlatır ve bir MCP istemcisini
taklit ederek (initialize → tools/list → tools/call) stdio üzerinden konuşur.
Bu, Antigravity/Hermes/Cursor gibi bir ajanın beyni nasıl mount edeceğinin
birebir aynısıdır — sadece ajan yerine biz konuşuyoruz.

python ornekler/mcp.py
"""

import json
import os
import subprocess
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mimar


def ciz(b=""):
    print("\n" + "=" * 70)
    if b:
        print(b); print("=" * 70)


def main():
    # 1) Bir beyin hazırla ve ona bir şey öğret (dosyaya)
    d = tempfile.mkdtemp()
    yol = os.path.join(d, "ajan.mimar")
    b = mimar.open(yol, engines=[mimar.engines.Echo()])
    b.declare(role="Proje yardımcı beyni.")
    b.teach("testleri nasıl koşarım", "python testler/test_cekirdek.py komutuyla.")
    b.close()

    ciz("1) Mimar MCP sunucusunu alt-süreç olarak başlat")
    proc = subprocess.Popen(
        [sys.executable, "-m", "mimar.mcp_server", yol],
        cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
    print("   başlatıldı:", " ".join(["python", "-m", "mimar.mcp_server",
                                       os.path.basename(yol)]))

    def gonder(obj, bekle=True):
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()
        if not bekle:
            return None
        line = proc.stdout.readline()
        return json.loads(line) if line.strip() else None

    try:
        ciz("2) initialize — el sıkışma (ajanın ilk yaptığı)")
        r = gonder({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05",
                               "capabilities": {}}})
        print("   sunucu:", r["result"]["serverInfo"],
              "proto", r["result"]["protocolVersion"])
        gonder({"jsonrpc": "2.0", "method": "notifications/initialized"},
               bekle=False)

        ciz("3) tools/list — ajan hangi araçları görüyor")
        r = gonder({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        for t in r["result"]["tools"]:
            print(f"   • {t['name']}: {t['description'][:60]}…")

        ciz("4) tools/call mimar_ask — ÖĞRETİLEN soru (derlenmiş, ucuz)")
        r = gonder({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "mimar_ask",
                               "arguments": {"query": "testleri nasıl koşarım"}}})
        print("  ", r["result"]["content"][0]["text"].replace("\n", " | "))

        ciz("5) tools/call mimar_ask — YENİ soru (modele düşer)")
        r = gonder({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                    "params": {"name": "mimar_ask",
                               "arguments": {"query": "bugün hava nasıl"}}})
        print("  ", r["result"]["content"][0]["text"].replace("\n", " | "))

        ciz("6) tools/call mimar_teach — ajan kendi tekrarını derliyor")
        r = gonder({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                    "params": {"name": "mimar_teach",
                               "arguments": {"prompt": "derleme komutu nedir",
                                             "answer": "make build"}}})
        print("  ", r["result"]["content"][0]["text"])
        r = gonder({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                    "params": {"name": "mimar_ask",
                               "arguments": {"query": "derleme komutu nedir"}}})
        print("   tekrar sor:", r["result"]["content"][0]["text"].split("\n")[0])

        ciz("7) resources/read mimar://stats — ajan beynin durumunu okuyor")
        r = gonder({"jsonrpc": "2.0", "id": 7, "method": "resources/read",
                    "params": {"uri": "mimar://stats"}})
        st = json.loads(r["result"]["contents"][0]["text"])
        print("   istatistik:", {k: st[k] for k in ("role", "cache", "runs")})

        ciz("8) bilinmeyen yöntem — düzgün JSON-RPC hatası")
        r = gonder({"jsonrpc": "2.0", "id": 8, "method": "yok/boyle"})
        print("   hata:", r["error"])

    finally:
        proc.stdin.close()
        proc.terminate()
        try:
            err = proc.stderr.read()
        except Exception:
            err = ""
        if err.strip():
            print("\n[sunucu stderr]:", err.strip()[:300])

    ciz("BİTTİ — bir ajan Mimar'ı tam olarak bu şekilde mount eder")


if __name__ == "__main__":
    main()
