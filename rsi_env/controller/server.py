"""Controlled interfaces: a token-authenticated local HTTP endpoint for the Host Controller.

The RSI Environment reaches controller services only through this endpoint;
the experimenter decides which services and which feedback are exposed per task
without changing the environment itself.
"""
from __future__ import annotations

import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .host import HostController, ServiceError


class ControllerServer:
    def __init__(self, host: HostController, port: int = 0):
        self.host = host
        self.token = secrets.token_urlsafe(24)
        token = self.token

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence default stderr logging
                pass

            def do_POST(self):  # noqa: N802
                if self.headers.get("Authorization") != f"Bearer {token}":
                    return self._send(401, {"error": "unauthorized"})
                name = self.path.strip("/").replace("-", "_")
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    payload = json.loads(self.rfile.read(n) or b"{}")
                    return self._send(200, host.call(name, payload))
                except ServiceError as e:
                    return self._send(400, {"error": str(e)})
                except TypeError as e:
                    return self._send(400, {"error": f"bad arguments: {e}"})
                except Exception as e:  # noqa: BLE001 - surface service errors as diagnostics
                    return self._send(500, {"error": f"{type(e).__name__}: {e}"})

            def _send(self, code, obj):
                body = json.dumps(obj, default=str).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
