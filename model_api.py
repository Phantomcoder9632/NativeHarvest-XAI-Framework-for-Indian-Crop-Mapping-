from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from ui_inference import discover_assets, load_model, predict


PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS = discover_assets(PROJECT_ROOT)
MODEL = load_model(ASSETS)


class PredictionHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/health"):
            self._send_json(
                {
                    "status": "ok",
                    "message": "Use POST /predict with JSON {'rows': [...]} to run inference.",
                    "model_path": str(ASSETS.model_path) if ASSETS.model_path else None,
                    "processed_dir": str(ASSETS.processed_dir) if ASSETS.processed_dir else None,
                    "architecture": ASSETS.architecture,
                    "sequence_length": ASSETS.sequence_length,
                }
            )
            return
        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        if self.path != "/predict":
            self._send_json({"error": "Not found"}, status=404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8"))
            frame = pd.DataFrame(payload["rows"])
            result = predict(frame, ASSETS, MODEL)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        response = {
            "top_crop": result["top_name"],
            "confidence": result["confidence"],
            "ranking": result["ranking"].to_dict(orient="records"),
            "sequence_length": ASSETS.sequence_length,
        }
        self._send_json(response)


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), PredictionHandler)
    print(f"Model API running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
