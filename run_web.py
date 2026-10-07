"""CloudOpt AI — Enterprise Web Server Runner.

Launches the FastAPI REST Server and Monochromatic Frontend Console.
Strictly Zero Fake Data.
"""
import argparse
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn


def open_browser(port: int):
    time.sleep(1.2)
    url = f"http://localhost:{port}"
    print(f"\n[INFO] Opening CloudOpt AI Monochromatic Console at: {url}\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="CloudOpt AI Web Console Runner")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind server (default: 0.0.0.0)")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")
    args = parser.parse_args()

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(args.port,), daemon=True).start()

    print(f"[INFO] Initializing CloudOpt AI Enterprise Server on {args.host}:{args.port}...")
    uvicorn.run("cba.api.server:app", host=args.host, port=args.port, reload=True)


if __name__ == "__main__":
    main()
