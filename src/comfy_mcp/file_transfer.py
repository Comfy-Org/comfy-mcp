"""MCP-side file transfer adapters for remote HTTP clients.

The engine remains ``comfy-cli``: inbound bytes are materialized only long
enough to become ordinary ``comfy upload`` path arguments, and completed
outputs are downloaded by ``comfy download`` before this module exposes them
through short-lived capability URLs on the MCP listener.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os
import secrets
import shlex
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlencode

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response

from . import argv
from .errors import ComfyCliError

# MCP SDK 2.0 caps a Streamable HTTP POST body at 4 MiB. Two MiB of decoded
# content expands to at most ~2.67 MiB of base64, leaving room for JSON-RPC,
# filenames, and the rest of the tool arguments without touching that ceiling.
_MAX_INLINE_UPLOAD_BYTES = 2 * 1024 * 1024
_MAX_INLINE_UPLOAD_BASE64_CHARS = 4 * ((_MAX_INLINE_UPLOAD_BYTES + 2) // 3)
_MAX_UPLOAD_NAME_BYTES = 255

_DOWNLOAD_TTL_SECONDS = 10 * 60
_DOWNLOAD_ROUTE = "/downloads/{token}/{filename}"
_DOWNLOAD_ROUTE_NAME = "comfy_mcp_download"


class UploadContent(BaseModel):
    """A file whose bytes travel inside a JSON MCP tool argument."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(
        description="Safe basename to give the staged file (for example input.webp)."
    )
    mime_type: str = Field(
        default="application/octet-stream",
        alias="mimeType",
        description="File MIME type metadata; the filename still controls ComfyUI use.",
        max_length=255,
    )
    data: str = Field(
        description="Base64-encoded file bytes (2 MiB decoded total per call).",
        max_length=_MAX_INLINE_UPLOAD_BASE64_CHARS,
        json_schema_extra={"contentEncoding": "base64"},
    )


@dataclass(frozen=True)
class PreparedUploads:
    """Concrete paths plus optional scratch ownership for one upload call."""

    directory: tempfile.TemporaryDirectory | None
    paths: list[str]
    display_by_path: dict[str, str]


def _upload_name(name: str, index: int) -> str:
    label = f"inline upload paths[{index}].name"
    if not name or name in {".", ".."}:
        raise ComfyCliError(f"invalid {label}: expected a non-empty filename")
    if "/" in name or "\\" in name or "\0" in name:
        raise ComfyCliError(
            f"invalid {label}: expected a basename without '/', '\\', or NUL"
        )
    try:
        encoded = os.fsencode(name)
    except UnicodeEncodeError as exc:
        raise ComfyCliError(f"invalid {label}: filename cannot be encoded") from exc
    if len(encoded) > _MAX_UPLOAD_NAME_BYTES:
        raise ComfyCliError(
            f"invalid {label}: encoded filename exceeds {_MAX_UPLOAD_NAME_BYTES} bytes"
        )
    return name


def _coerce_upload_content(value: Any, index: int) -> UploadContent:
    if isinstance(value, UploadContent):
        return value
    if not isinstance(value, dict):
        raise ComfyCliError(
            f"invalid paths[{index}]: expected a string path or inline file object"
        )
    try:
        return UploadContent.model_validate(value)
    except ValidationError as exc:
        # Pydantic's rendered error can include ``data`` verbatim. Never place a
        # caller's base64 payload in an MCP error or the opt-in failure log.
        raise ComfyCliError(
            f"invalid inline upload paths[{index}]: expected name, optional "
            "mimeType, and valid base64 data within the 2 MiB call limit"
        ) from exc


def _decode_upload_data(content: UploadContent, index: int) -> bytes:
    try:
        return base64.b64decode(content.data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ComfyCliError(
            f"invalid inline upload paths[{index}].data: expected strict base64"
        ) from exc


def _write_owner_only(path: str, data: bytes) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)


def prepare_uploads(sources: Any) -> PreparedUploads:
    """Validate/materialize local paths and inline objects without network I/O."""

    if not isinstance(sources, list):
        # Preserve the established error wording/order for a bare string and
        # every other wrong container type.
        argv._validate_upload_paths(sources)
        raise AssertionError("upload path validation unexpectedly returned")
    if not sources:
        argv._validate_upload_paths([])
        raise AssertionError("empty upload path validation unexpectedly returned")
    if all(isinstance(source, str) for source in sources):
        paths = list(sources)
        argv._validate_upload_paths(paths)
        return PreparedUploads(None, paths, {})

    # Refuse an oversized batch before decoding or touching disk. Reusing the
    # established list validator keeps the public count error byte-identical.
    if len(sources) > argv._MAX_UPLOAD_PATHS:
        argv._validate_upload_paths(["x"] * len(sources))

    directory = tempfile.TemporaryDirectory(prefix="comfy-mcp-upload-")
    paths: list[str] = []
    display_by_path: dict[str, str] = {}
    names: set[str] = set()
    total_bytes = 0
    try:
        for index, source in enumerate(sources):
            if isinstance(source, str):
                paths.append(source)
                continue
            content = _coerce_upload_content(source, index)
            name = _upload_name(content.name, index)
            name_key = os.path.normcase(name)
            if name_key in names:
                raise ComfyCliError(
                    f"invalid inline upload paths[{index}].name: duplicate filename "
                    f"{name!r} in one call"
                )
            names.add(name_key)
            data = _decode_upload_data(content, index)
            total_bytes += len(data)
            if total_bytes > _MAX_INLINE_UPLOAD_BYTES:
                raise ComfyCliError(
                    "invalid inline uploads: decoded content exceeds the 2 MiB "
                    "total per call (the MCP HTTP request limit is 4 MiB)"
                )
            path = os.path.join(directory.name, name)
            try:
                _write_owner_only(path, data)
            except OSError as exc:
                raise ComfyCliError(
                    f"invalid inline upload paths[{index}].name: the MCP server "
                    "cannot stage this filename"
                ) from exc
            paths.append(path)
            display_by_path[path] = name

        argv._validate_upload_paths(paths)
        return PreparedUploads(directory, paths, display_by_path)
    except BaseException:
        directory.cleanup()
        raise


def rewrite_upload_result(data: Any, display_by_path: dict[str, str]) -> Any:
    """Replace only scratch-path echoes with the caller-visible inline name."""

    if not display_by_path or not isinstance(data, dict):
        return data
    uploads = data.get("uploads")
    if not isinstance(uploads, list):
        return data
    rewritten = dict(data)
    rows: list[Any] = []
    for upload in uploads:
        if not isinstance(upload, dict):
            rows.append(upload)
            continue
        row = dict(upload)
        local_path = row.get("local_path")
        if isinstance(local_path, str) and local_path in display_by_path:
            row["local_path"] = display_by_path[local_path]
        rows.append(row)
    rewritten["uploads"] = rows
    return rewritten


@dataclass
class _DownloadBatch:
    directory: tempfile.TemporaryDirectory
    expires: int
    tokens: set[str] = field(default_factory=set)
    active: int = 0
    expired: bool = False
    cleaned: bool = False
    timer: threading.Timer | None = None


@dataclass(frozen=True)
class _DownloadEntry:
    path: str
    filename: str
    batch: _DownloadBatch


@dataclass(frozen=True)
class _DownloadLease:
    entry: _DownloadEntry


class _SignedDownloadStore:
    """Process-local, bounded-lifetime capability URLs for downloaded outputs."""

    def __init__(self, ttl_seconds: int = _DOWNLOAD_TTL_SECONDS) -> None:
        self._ttl_seconds = ttl_seconds
        self._secret = secrets.token_bytes(32)
        self._entries: dict[str, _DownloadEntry] = {}
        self._lock = threading.Lock()

    def _signature(self, token: str, filename: str, expires: int) -> str:
        payload = f"{token}\n{filename}\n{expires}".encode()
        return hmac.new(self._secret, payload, hashlib.sha256).hexdigest()

    @staticmethod
    def _client_path(out_dir: str, filename: str) -> str:
        separator = "\\" if "\\" in out_dir and "/" not in out_dir else "/"
        base = out_dir.rstrip("/\\")
        return f"{base}{separator}{filename}" if base else filename

    @staticmethod
    def _windows_quote(value: str) -> str:
        # The generated URL contains ``&`` and therefore must be quoted for
        # cmd.exe even though it has no whitespace. Windows filenames cannot
        # contain a double quote; doubling one still keeps this helper safe for
        # a malformed caller-provided output directory.
        return f'"{value.replace(chr(34), chr(34) * 2)}"'

    @staticmethod
    def _cleanup_batch(batch: _DownloadBatch) -> None:
        if batch.cleaned:
            return
        batch.cleaned = True
        batch.directory.cleanup()

    def _expire(self, batch: _DownloadBatch) -> None:
        cleanup = False
        with self._lock:
            if batch.expired:
                return
            batch.expired = True
            for token in batch.tokens:
                self._entries.pop(token, None)
            cleanup = batch.active == 0
        if cleanup:
            self._cleanup_batch(batch)

    def publish(
        self,
        directory: tempfile.TemporaryDirectory,
        data: Any,
        *,
        base_url: str,
        client_out_dir: str,
    ) -> dict[str, Any]:
        """Own a completed ``comfy download`` directory and publish its files."""

        if not isinstance(data, dict) or not isinstance(data.get("files"), list):
            directory.cleanup()
            raise ComfyCliError(
                "comfy-cli download returned no file list for remote MCP transfer"
            )

        root = os.path.realpath(directory.name)
        rows: list[tuple[dict[str, Any], str, str]] = []
        for index, raw in enumerate(data["files"]):
            if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
                directory.cleanup()
                raise ComfyCliError(
                    f"comfy-cli download returned an invalid files[{index}] entry"
                )
            reported_path = raw["path"]
            path = os.path.realpath(
                reported_path
                if os.path.isabs(reported_path)
                else os.path.join(root, reported_path)
            )
            if not (
                path == root or path.startswith(root + os.sep)
            ) or not os.path.isfile(path):
                directory.cleanup()
                raise ComfyCliError(
                    "comfy-cli download reported a file outside its remote-transfer "
                    f"scratch directory at files[{index}]"
                )
            filename = os.path.basename(path)
            rows.append((dict(raw), path, filename))

        expires = int(time.time()) + self._ttl_seconds
        batch = _DownloadBatch(directory=directory, expires=expires)
        published: list[dict[str, Any]] = []
        with self._lock:
            for row, path, filename in rows:
                token = secrets.token_urlsafe(32)
                signature = self._signature(token, filename, expires)
                query = urlencode({"expires": expires, "signature": signature})
                url = f"{base_url.rstrip('/')}/downloads/{token}/{quote(filename)}?{query}"
                client_path = self._client_path(client_out_dir, filename)
                row["path"] = client_path
                row["url"] = url
                row["expires_at"] = expires
                row["command"] = " ".join(
                    shlex.quote(part)
                    for part in (
                        "curl",
                        "--fail",
                        "--location",
                        "--output",
                        client_path,
                        url,
                    )
                )
                row["windows_command"] = " ".join(
                    (
                        "curl.exe",
                        "--fail",
                        "--location",
                        "--output",
                        self._windows_quote(client_path),
                        self._windows_quote(url),
                    )
                )
                batch.tokens.add(token)
                self._entries[token] = _DownloadEntry(path, filename, batch)
                published.append(row)

        if batch.tokens:
            timer = threading.Timer(self._ttl_seconds + 1, self._expire, args=(batch,))
            timer.daemon = True
            batch.timer = timer
            timer.start()
        else:
            self._cleanup_batch(batch)

        result = dict(data)
        result["out_dir"] = client_out_dir
        result["files"] = published
        result["download_url_ttl_seconds"] = self._ttl_seconds
        return result

    def acquire(
        self, token: str, filename: str, expires: int, signature: str
    ) -> _DownloadLease | None:
        expected = self._signature(token, filename, expires)
        if not hmac.compare_digest(expected, signature):
            return None
        cleanup: _DownloadBatch | None = None
        with self._lock:
            entry = self._entries.get(token)
            if (
                entry is None
                or entry.filename != filename
                or entry.batch.expires != expires
                or expires < int(time.time())
            ):
                if entry is not None and expires < int(time.time()):
                    cleanup = entry.batch
                lease = None
            else:
                entry.batch.active += 1
                lease = _DownloadLease(entry)
        if cleanup is not None:
            self._expire(cleanup)
        return lease

    def release(self, lease: _DownloadLease) -> None:
        cleanup = False
        batch = lease.entry.batch
        with self._lock:
            batch.active = max(0, batch.active - 1)
            cleanup = batch.expired and batch.active == 0
        if cleanup:
            self._cleanup_batch(batch)

    def close(self) -> None:
        """Test/process cleanup hook for every outstanding signed download."""

        with self._lock:
            batches = {id(entry.batch): entry.batch for entry in self._entries.values()}
        for batch in batches.values():
            if batch.timer is not None:
                batch.timer.cancel()
            self._expire(batch)


_DOWNLOAD_STORE = _SignedDownloadStore()


def publish_downloads(
    directory: tempfile.TemporaryDirectory,
    data: Any,
    *,
    base_url: str,
    client_out_dir: str,
) -> dict[str, Any]:
    return _DOWNLOAD_STORE.publish(
        directory,
        data,
        base_url=base_url,
        client_out_dir=client_out_dir,
    )


async def serve_signed_download(request: Request) -> Response:
    """Serve one signed output without exposing the scratch filesystem path."""

    token = request.path_params.get("token", "")
    filename = request.path_params.get("filename", "")
    try:
        expires = int(request.query_params.get("expires", ""))
    except ValueError:
        expires = 0
    signature = request.query_params.get("signature", "")
    lease = _DOWNLOAD_STORE.acquire(token, filename, expires, signature)
    if lease is None:
        return JSONResponse(
            {"error": "download link is invalid or expired"},
            status_code=404,
            headers={"Cache-Control": "no-store"},
        )
    return FileResponse(
        lease.entry.path,
        filename=lease.entry.filename,
        headers={
            "Cache-Control": "private, no-store",
            "Content-Security-Policy": "default-src 'none'",
            "X-Content-Type-Options": "nosniff",
        },
        background=BackgroundTask(_DOWNLOAD_STORE.release, lease),
    )
