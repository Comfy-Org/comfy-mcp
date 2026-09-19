"""MCP-side file transfer adapters for remote HTTP clients.

The engine remains ``comfy-cli``.  A remote caller receives a short-lived
single-use upload command; the corresponding route materializes those bytes
only long enough to pass a normal path to ``comfy upload``.  Completed outputs
take the inverse path: ``comfy download`` writes owner-only scratch and this
module publishes short-lived capability URLs on the same MCP listener.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import secrets
import shlex
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO, Literal
from urllib.parse import quote, urlencode

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.types import Receive, Scope, Send

from . import argv
from .errors import ComfyCliError

_MAX_UPLOAD_NAME_BYTES = 255
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_UPLOAD_TTL_SECONDS = 5 * 60
_UPLOAD_ROUTE = "/api/uploads/{token}"
_UPLOAD_ROUTE_NAME = "comfy_mcp_upload"
_UPLOAD_MIME_TYPES = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

_DOWNLOAD_TTL_SECONDS = 5 * 60
_DOWNLOAD_ROUTE = "/downloads/{token}/{filename}"
_DOWNLOAD_ROUTE_NAME = "comfy_mcp_download"

ClientOS = Literal["darwin", "linux", "windows"]


def _posix_quote(value: str) -> str:
    """Always single-quote one shell argument, matching ComfyCloud's command."""

    return "'" + value.replace("'", "'\"'\"'") + "'"


def _open_owner_only(path: str) -> BinaryIO:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    return os.fdopen(fd, "wb")


@dataclass(frozen=True, slots=True)
class UploadRequest:
    """Validated client-local path metadata; no file access occurs here."""

    file_path: str
    filename: str
    mime_type: str
    client_os: ClientOS


@dataclass(frozen=True, slots=True)
class _UploadEntry:
    filename: str
    mime_type: str
    expires: int


def prepare_upload_request(file_path: str, client_os: ClientOS) -> UploadRequest:
    """Validate the Cloud-compatible upload arguments without opening the path."""

    argv._guard_arg_len("file_path", file_path)
    file_path = argv._reject_nul("file_path", file_path)
    if client_os == "windows":
        path = PureWindowsPath(file_path)
    elif client_os in {"darwin", "linux"}:
        path = PurePosixPath(file_path)
    else:
        raise ComfyCliError(
            "invalid client_os: expected one of 'darwin', 'linux', or 'windows'"
        )
    if not path.is_absolute() or not path.name:
        raise ComfyCliError("file_path must be an absolute path to an image file")
    try:
        encoded_name = os.fsencode(path.name)
    except UnicodeEncodeError as exc:
        raise ComfyCliError("file_path filename cannot be encoded") from exc
    if len(encoded_name) > _MAX_UPLOAD_NAME_BYTES:
        raise ComfyCliError(
            f"file_path filename exceeds {_MAX_UPLOAD_NAME_BYTES} encoded bytes"
        )
    mime_type = _UPLOAD_MIME_TYPES.get(path.suffix.lower())
    if mime_type is None:
        raise ComfyCliError(
            "upload_file accepts image files ending in .jpg, .jpeg, .png, .webp, "
            "or .gif"
        )
    return UploadRequest(file_path, path.name, mime_type, client_os)


class _UploadStore:
    """Small process-local registry for single-use upload capability URLs."""

    def __init__(self, ttl_seconds: int = _UPLOAD_TTL_SECONDS) -> None:
        self._ttl_seconds = ttl_seconds
        self._entries: dict[str, _UploadEntry] = {}
        self._lock = threading.Lock()

    def mint(self, request: UploadRequest, *, base_url: str) -> str:
        now = int(time.time())
        token = secrets.token_urlsafe(24)
        with self._lock:
            self._entries = {
                key: value
                for key, value in self._entries.items()
                if value.expires >= now
            }
            self._entries[token] = _UploadEntry(
                request.filename,
                request.mime_type,
                now + self._ttl_seconds,
            )
        url = f"{base_url.rstrip('/')}{_UPLOAD_ROUTE.format(token=token)}"
        if request.client_os == "windows":
            command = " ".join(
                (
                    "curl.exe",
                    "-sS",
                    "--fail-with-body",
                    "-X",
                    "PUT",
                    "-H",
                    f'"Content-Type: {request.mime_type}"',
                    "--upload-file",
                    _SignedDownloadStore._windows_quote(request.file_path),
                    "--",
                    _SignedDownloadStore._windows_quote(url),
                )
            )
            shell = "PowerShell"
        else:
            command = (
                "curl -sS --fail-with-body -X PUT "
                f"-H {_posix_quote(f'Content-Type: {request.mime_type}')} "
                f"--upload-file {_posix_quote(request.file_path)} -- "
                f"{_posix_quote(url)}"
            )
            shell = "Bash"
        return (
            f"Run this command via {shell} to upload the file. It is "
            "credential-free — the single-use URL is the authorization — so run "
            "it exactly as emitted; do NOT add any Authorization header or "
            f"credential.\n\n{command}\n\nOn success it prints a JSON body like "
            '{"name":"...","subfolder":"","type":"input"} — use that '
            '"name" as the "image" value in your workflow\'s LoadImage node; no '
            "further upload_file call is needed. The URL is single-use with a "
            "short TTL: if the command fails (404 = used/expired), re-invoke "
            "upload_file for a fresh URL instead of re-running this one. Images "
            "must be at most 50 MiB — a larger file returns 413 and consumes the "
            "URL, so shrink it first rather than retrying. The PUT targets the "
            "same MCP host as this session."
        )

    def consume(self, token: str) -> _UploadEntry | None:
        now = int(time.time())
        with self._lock:
            entry = self._entries.pop(token, None)
        if entry is None or entry.expires < now:
            return None
        return entry

    def close(self) -> None:
        with self._lock:
            self._entries.clear()


_UPLOAD_STORE = _UploadStore()


def mint_upload_command(request: UploadRequest, *, base_url: str) -> str:
    """Mint the same command-oriented remote upload response as ComfyCloud."""

    return _UPLOAD_STORE.mint(request, base_url=base_url)


def uploaded_name(data: Any) -> str:
    """Read the uploaded input name from comfy-cli's own envelope data."""

    if isinstance(data, dict):
        direct = data.get("name")
        if isinstance(direct, str) and direct:
            return direct
        uploads = data.get("uploads")
        if isinstance(uploads, list) and uploads and isinstance(uploads[0], dict):
            for key in ("cloud_name", "name", "filename"):
                value = uploads[0].get(key)
                if isinstance(value, str) and value:
                    return value
    raise ComfyCliError("comfy-cli upload returned no uploaded input name")


async def receive_upload(
    request: Request,
    uploader: Callable[[str], Awaitable[Any]],
) -> Response:
    """Consume one capability URL, then forward its bytes through comfy-cli."""

    entry = _UPLOAD_STORE.consume(request.path_params.get("token", ""))
    if entry is None:
        return JSONResponse(
            {"error": "upload URL is invalid, used, or expired"},
            status_code=404,
            headers={"Cache-Control": "no-store"},
        )
    directory = tempfile.TemporaryDirectory(prefix="comfy-mcp-upload-")
    path = os.path.join(directory.name, entry.filename)
    try:
        handle = await asyncio.to_thread(_open_owner_only, path)
        total = 0
        try:
            async for chunk in request.stream():
                total += len(chunk)
                if total > _MAX_UPLOAD_BYTES:
                    return JSONResponse(
                        {"error": "file exceeds the 50 MiB upload limit"},
                        status_code=413,
                        headers={"Cache-Control": "no-store"},
                    )
                if chunk:
                    await asyncio.to_thread(handle.write, chunk)
        finally:
            await asyncio.to_thread(handle.close)
        data = await uploader(path)
        return JSONResponse(
            {"name": uploaded_name(data), "subfolder": "", "type": "input"},
            headers={"Cache-Control": "no-store"},
        )
    except ComfyCliError as exc:
        return JSONResponse(
            {"error": str(exc)},
            status_code=502,
            headers={"Cache-Control": "no-store"},
        )
    finally:
        await asyncio.to_thread(directory.cleanup)


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
        client_os: ClientOS,
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
                route = _DOWNLOAD_ROUTE.format(
                    token=token,
                    filename=quote(filename),
                )
                url = f"{base_url.rstrip('/')}{route}?{query}"
                client_path = self._client_path(client_out_dir, filename)
                row["path"] = client_path
                row["url"] = url
                row["download_url"] = url
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
                row["download_command"] = (
                    row["windows_command"] if client_os == "windows" else row["command"]
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
        result["download_command"] = " && ".join(
            row["download_command"] for row in published
        )
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


class _LeasedFileResponse(FileResponse):
    """Release one download lease whenever the ASGI response exits."""

    def __init__(
        self,
        lease: _DownloadLease,
        store: _SignedDownloadStore,
    ) -> None:
        self._lease = lease
        self._store = store
        super().__init__(
            lease.entry.path,
            filename=lease.entry.filename,
            headers={
                "Cache-Control": "private, no-store",
                "Content-Security-Policy": "default-src 'none'",
                "X-Content-Type-Options": "nosniff",
            },
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Starlette background tasks run only after the response body finishes.
            # A send error or task cancellation can exit earlier, so the lease
            # belongs in a real finally block rather than a BackgroundTask.
            self._store.release(self._lease)


def publish_downloads(
    directory: tempfile.TemporaryDirectory,
    data: Any,
    *,
    base_url: str,
    client_out_dir: str,
    client_os: ClientOS,
) -> dict[str, Any]:
    return _DOWNLOAD_STORE.publish(
        directory,
        data,
        base_url=base_url,
        client_out_dir=client_out_dir,
        client_os=client_os,
    )


def download_message(data: dict[str, Any], *, client_os: ClientOS) -> str:
    """Render the Cloud-style URL + ready command response for MCP content."""

    files = data.get("files", [])
    urls = [row.get("download_url") for row in files if isinstance(row, dict)]
    links = "\n".join(f"- {url}" for url in urls if isinstance(url, str))
    shell = "PowerShell" if client_os == "windows" else "Bash"
    return (
        "Output is ready. Temporary download URL(s):\n\n"
        f"{links}\n\nRun this command via {shell} to download the output. "
        "Run it exactly as emitted; do not edit or re-encode the signed URL.\n\n"
        f"{data.get('download_command', '')}\n\nThe URL is valid for a short "
        "window. If it expires, call fetch_outputs again for a fresh URL."
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
    return _LeasedFileResponse(lease, _DOWNLOAD_STORE)
