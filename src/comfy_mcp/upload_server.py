"""Loopback HTTP service that accepts one streaming PUT per upload session.

Run it beside the stdio MCP server::

    comfy-mcp-upload-server

It binds ``127.0.0.1:8192`` unless ``COMFY_MCP_UPLOAD_HOST`` /
``COMFY_MCP_UPLOAD_PORT`` say otherwise, and only on loopback. The public
name is the reverse proxy in front of that bind. Binary bodies are written
by :mod:`comfy_mcp.upload_session`; this module does not call ComfyUI.

Routes: ``PUT /upload/{upload_id}`` and ``GET /healthz``.
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from . import cli
from .errors import ComfyCliError
from .upload_session import UploadRejected, accept_put, bind_host_port

_LOG = logging.getLogger("comfy_mcp.upload_server")


class UploadHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "comfy-mcp-upload"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(3600)

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path == "/healthz" and not parsed.query and not parsed.fragment:
            self._send(200, b"ok\n", "text/plain; charset=utf-8")
            return
        self._send(404, b'{"error":"not found"}\n', "application/json")

    def do_PUT(self) -> None:
        upload_id = _path_upload_id(self.path)
        try:
            self._put()
        except UploadRejected as exc:
            self._send(exc.status, _error_body(exc), "application/json")
        except Exception as exc:  # noqa: BLE001 - the client gets a fixed message, not the traceback
            _LOG.exception(
                "upload_id=%s stage=put_request state=- error_type=%s",
                upload_id or "-",
                type(exc).__name__,
            )
            self._send(
                500,
                _error_body(
                    UploadRejected(
                        500, "upload failed.", upload_id=upload_id, stage="put_request"
                    )
                ),
                "application/json",
            )

    def _put(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.query or parsed.fragment:
            raise UploadRejected(404, "unknown upload.")
        parts = parsed.path.split("/")
        if len(parts) != 3 or parts[0] != "" or parts[1] != "upload":
            raise UploadRejected(404, "unknown upload.")
        result = accept_put(
            parts[2],
            self.headers.get("Authorization"),
            self.headers.get("Content-Length"),
            self.headers.get("Content-Type"),
            self.rfile.read,
        )
        body = json.dumps(
            {"ok": True, "byte_size": result["byte_size"], "sha256": result["sha256"]}
        ).encode("utf-8")
        self._send(200, body, "application/json")

    def log_message(self, fmt: str, *args: Any) -> None:
        code = args[1] if len(args) > 1 else "-"
        upload_id = _path_upload_id(self.path)
        _LOG.info(
            "upload_id=%s stage=http http_status=%s",
            upload_id or "-",
            code,
        )

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


def _path_upload_id(path: str) -> str | None:
    parsed = urlsplit(path)
    parts = parsed.path.split("/")
    if len(parts) == 3 and parts[0] == "" and parts[1] == "upload" and parts[2]:
        return parts[2]
    return None


def _error_body(exc: UploadRejected) -> bytes:
    payload = {"error": exc.message, "stage": exc.stage}
    if exc.upload_id:
        payload["upload_id"] = exc.upload_id
    return (json.dumps(payload) + "\n").encode("utf-8")


def serve(host: str, port: int) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "::1"}:
        raise ComfyCliError(
            "the upload service only binds loopback. Publish it through the reverse proxy."
        )
    httpd = ThreadingHTTPServer((host, port), UploadHandler)
    httpd.daemon_threads = True
    return httpd


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    cli._log_startup("comfy-mcp-upload-server")
    host, port = bind_host_port()
    httpd = serve(host, port)
    _LOG.info("upload server listening on %s:%s", host, port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()
