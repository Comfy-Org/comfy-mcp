"""ChatGPT host-file ingress for ``init_upload``.

This is the only HTTP client in the MCP process. It downloads the ``download_url``
on an OpenAI file parameter and writes the original bytes to a temporary file.
It is not a general URL tool, and it never calls ComfyUI.

The URL, its query string, and the file bytes are not logged.
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urljoin, urlsplit

from pydantic import BaseModel, ConfigDict

from .errors import ComfyCliError
from .upload_session import check_mime, max_upload_bytes, payload_path, safe_filename

_CHUNK = 64 * 1024
_CONNECT_TIMEOUT = 30.0
_OVERALL_SECONDS = 3600.0
_MAX_REDIRECTS = 5
_MAX_URL_CHARS = 8192
_BLOCKED_HOSTS = frozenset({"localhost", "localhost.localdomain"})


class OpenAIFile(BaseModel):
    """The host file object ChatGPT passes when ``openai/fileParams`` names ``file``.

    ``mime_type`` and ``file_name`` are optional. The JSON Schema advertised to
    the client declares all four properties as strings and requires only
    ``download_url`` and ``file_id``.
    """

    model_config = ConfigDict(extra="forbid")

    download_url: str
    file_id: str
    mime_type: str | None = None
    file_name: str | None = None

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: Any, handler: Any
    ) -> dict[str, Any]:
        schema = handler(core_schema)
        schema["type"] = "object"
        schema["additionalProperties"] = False
        schema["properties"] = {
            "download_url": {"type": "string"},
            "file_id": {"type": "string"},
            "mime_type": {"type": "string"},
            "file_name": {"type": "string"},
        }
        schema["required"] = ["download_url", "file_id"]
        schema.pop("title", None)
        schema.pop("description", None)
        return schema


class StagedHostFile:
    """Original bytes staged from a host file parameter."""

    def __init__(
        self, path: str, filename: str, mime_type: str, byte_size: int, sha256: str
    ) -> None:
        self.path = path
        self.filename = filename
        self.mime_type = mime_type
        self.byte_size = byte_size
        self.sha256 = sha256


class _SeenRedirect(Exception):
    def __init__(self, location: str) -> None:
        super().__init__("redirect")
        self.location = location


class _RedirectStop(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        raise _SeenRedirect(newurl if isinstance(newurl, str) else "")


def stage_openai_file(file: OpenAIFile, dest_dir: str) -> StagedHostFile:
    """Stream ``file.download_url`` into ``dest_dir`` and return the staged path."""
    if not isinstance(file, OpenAIFile):
        raise ComfyCliError("the host file parameter is not usable.")
    filename = safe_filename(
        file.file_name if isinstance(file.file_name, str) else "",
        fallback="upload.bin",
    )
    dest = payload_path(dest_dir, filename)
    if os.path.basename(dest) != filename:
        raise ComfyCliError("the host file could not be staged.")
    byte_size, digest = _download(file.download_url, dest)
    return StagedHostFile(
        path=dest,
        filename=filename,
        mime_type=_metadata_mime(file.mime_type),
        byte_size=byte_size,
        sha256=digest,
    )


def _metadata_mime(value: str | None) -> str:
    if not isinstance(value, str):
        return "application/octet-stream"
    try:
        return check_mime(value)
    except ComfyCliError:
        return "application/octet-stream"


def _download(url: str, dest: str) -> tuple[int, str]:
    if not isinstance(url, str) or not url or len(url) > _MAX_URL_CHARS:
        raise ComfyCliError("the host file URL is not an allowed public https address.")
    deadline = time.monotonic() + _OVERALL_SECONDS
    current = url
    hops = 0
    while True:
        _assert_public(current)
        if time.monotonic() > deadline:
            raise ComfyCliError("the host file could not be downloaded.")
        try:
            response = _open_once(current, _CONNECT_TIMEOUT)
            break
        except _SeenRedirect as redirect:
            hops += 1
            if hops > _MAX_REDIRECTS or not redirect.location:
                raise ComfyCliError("the host file could not be downloaded.") from None
            current = urljoin(current, redirect.location)
        except (urllib.error.URLError, OSError, ValueError):
            raise ComfyCliError("the host file could not be downloaded.") from None
    try:
        return _write_response(response, dest, deadline)
    finally:
        response.close()


def _open_once(url: str, timeout: float) -> Any:
    request = urllib.request.Request(url, method="GET")
    opener = urllib.request.build_opener(_RedirectStop, urllib.request.ProxyHandler({}))
    return opener.open(request, timeout=timeout)


def _write_response(response: Any, dest: str, deadline: float) -> tuple[int, str]:
    status = getattr(response, "status", None) or getattr(response, "code", None)
    if status != 200:
        raise ComfyCliError("the host file could not be downloaded.")
    limit = max_upload_bytes()
    declared = (
        response.headers.get("Content-Length") if response.headers is not None else None
    )
    if declared is not None:
        try:
            announced = int(declared)
        except (TypeError, ValueError):
            raise ComfyCliError("the host file could not be downloaded.") from None
        if announced < 0 or announced > limit:
            raise ComfyCliError("the host file exceeds COMFY_MCP_MAX_UPLOAD_MB.")
    hasher = hashlib.sha256()
    written = 0
    fd = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.chmod(dest, 0o600)
        while True:
            if time.monotonic() > deadline:
                raise ComfyCliError("the host file could not be downloaded.")
            chunk = response.read(_CHUNK)
            if not chunk:
                break
            written += len(chunk)
            if written > limit:
                raise ComfyCliError("the host file exceeds COMFY_MCP_MAX_UPLOAD_MB.")
            hasher.update(chunk)
            view = memoryview(chunk)
            while view:
                n = os.write(fd, view)
                view = view[n:]
        os.fsync(fd)
    except Exception:
        os.close(fd)
        try:
            os.unlink(dest)
        except FileNotFoundError:
            pass
        raise
    else:
        os.close(fd)
    if written < 1:
        try:
            os.unlink(dest)
        except FileNotFoundError:
            pass
        raise ComfyCliError("the host file could not be downloaded.")
    if declared is not None and written != int(declared):
        try:
            os.unlink(dest)
        except FileNotFoundError:
            pass
        raise ComfyCliError("the host file could not be downloaded.")
    return written, hasher.hexdigest()


def _assert_public(url: str) -> None:
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        raise ComfyCliError(
            "the host file URL is not an allowed public https address."
        ) from None
    if (
        parts.scheme != "https"
        or parts.username
        or parts.password
        or not host
        or _name_blocked(host)
    ):
        raise ComfyCliError("the host file URL is not an allowed public https address.")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if _ip_blocked(literal):
            raise ComfyCliError(
                "the host file URL is not an allowed public https address."
            )
        return
    port = parts.port or 443
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        raise ComfyCliError("the host file could not be downloaded.") from None
    if not infos:
        raise ComfyCliError("the host file URL is not an allowed public https address.")
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            raise ComfyCliError(
                "the host file URL is not an allowed public https address."
            ) from None
        if _ip_blocked(ip):
            raise ComfyCliError(
                "the host file URL is not an allowed public https address."
            )


def _name_blocked(host: str) -> bool:
    name = host.lower().rstrip(".")
    if name in _BLOCKED_HOSTS or name.endswith(".localhost") or name.endswith(".local"):
        return True
    # Decimal or octal IP spellings are not a hostname we will resolve.
    return bool(name) and name.replace(".", "").isdigit()


def _ip_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return _ip_blocked(ip.ipv4_mapped)
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
        or ip.is_reserved
    )
