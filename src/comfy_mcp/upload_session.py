"""Shared upload spool for ``init_upload`` / the upload HTTP service.

The MCP process and ``comfy-mcp-upload-server`` are separate processes, so a
session lives entirely on disk under ``COMFY_MCP_UPLOAD_DIR``. No database.

State machine (one streaming PUT, no multipart in V1)::

    initialized → uploading → ready → completing → completed

``completed`` removes the directory. A failed PUT returns to ``initialized``
so the same one-time token can be retried until it expires. A failed
``complete_upload`` returns to ``ready`` and records ``last_error``. Expired
sessions are removed opportunistically; a session left in ``completing`` is
kept until its (extended) expiry so an in-flight ``comfy upload`` is not
deleted underneath itself.

Bytes land in ``payload.part`` only while the PUT is in progress. After the
size and SHA-256 checks, that file is renamed to the session's
``safe_filename`` and only then is the state set to ``ready``. ``comfy upload``
receives that named path, so ComfyUI does not see ``payload.part``.

A future resumable upload can add a ``parts`` list to the manifest. V1 does
not implement that.

The client never chooses this directory. The one-time token is returned to
the caller once and stored here only as a SHA-256 hash.
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import shutil
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from .errors import ComfyCliError

_LOG = logging.getLogger("comfy_mcp.upload")

_PUT_STAGE = {
    400: "put_body",
    401: "put_auth",
    404: "put_lookup",
    409: "put_state",
    411: "put_headers",
    413: "put_limit",
    415: "put_type",
    500: "put_store",
}


def _event(
    stage: str,
    upload_id: str | None,
    state: str | None = None,
    **fields: object,
) -> None:
    """One correlation line. Callers pass only values that are safe to log."""
    parts = [f"upload_id={upload_id or '-'}", f"stage={stage}"]
    if state is not None:
        parts.append(f"state={state}")
    for key, value in fields.items():
        rendered = _log_value(value)
        if rendered is None:
            continue
        parts.append(f"{key}={rendered}")
    _LOG.info(" ".join(parts))


def _log_value(value: object) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return None
    text = str(value)
    if any(char in text for char in "\r\n\x00"):
        return None
    lowered = text.lower()
    if "bearer " in lowered or "authorization" in lowered or "://" in lowered:
        return None
    if len(text) > 200:
        text = text[:200]
    return text


_DEFAULT_SPOOL = "/var/lib/comfy-mcp/uploads"
_DEFAULT_TTL_SECONDS = 600
_DEFAULT_MAX_UPLOAD_MB = 1024
_MAX_UPLOAD_MB_CEILING = 8192
_MIN_TTL_SECONDS = 30
_MAX_TTL_SECONDS = 24 * 60 * 60
_CHUNK = 64 * 1024
_MAX_FILENAME_BYTES = 255
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,80}$")
_DEVICE_RE = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\.|$)", re.IGNORECASE)
_ORIGIN_RE = re.compile(r"https://[A-Za-z0-9.-]+(?::[0-9]{1,5})?\Z")
_MIME_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]{0,126}/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]{0,126}\Z"
)


class UploadRejected(Exception):
    """A PUT the upload service should reject. ``message`` is client-safe."""

    def __init__(
        self,
        status: int,
        message: str,
        *,
        stage: str = "put",
        upload_id: str | None = None,
        state: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.stage = stage
        self.upload_id = upload_id
        self.state = state


class UploadFailed(Exception):
    """An expected upload failure the MCP tool returns as ``upload_error``."""

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        upload_id: str | None = None,
        state: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.upload_id = upload_id
        self.state = state
        self.retryable = retryable


class ReadyUpload:
    """A session ``complete_upload`` may hand to ``comfy upload``."""

    def __init__(
        self,
        upload_id: str,
        path: str,
        filename: str,
        mime_type: str,
        byte_size: int,
        sha256: str,
        overwrite: bool,
    ) -> None:
        self.upload_id = upload_id
        self.path = path
        self.filename = filename
        self.mime_type = mime_type
        self.byte_size = byte_size
        self.sha256 = sha256
        self.overwrite = overwrite


def spool_dir() -> str:
    raw = os.environ.get("COMFY_MCP_UPLOAD_DIR", _DEFAULT_SPOOL)
    if not raw or "\x00" in raw:
        raise ComfyCliError("COMFY_MCP_UPLOAD_DIR is not a usable path.")
    return raw


def max_upload_bytes() -> int:
    raw = os.environ.get("COMFY_MCP_MAX_UPLOAD_MB", str(_DEFAULT_MAX_UPLOAD_MB))
    try:
        mb = int(raw)
    except (TypeError, ValueError):
        mb = _DEFAULT_MAX_UPLOAD_MB
    if mb < 1:
        mb = 1
    if mb > _MAX_UPLOAD_MB_CEILING:
        mb = _MAX_UPLOAD_MB_CEILING
    return mb * 1024 * 1024


def ttl_seconds() -> int:
    raw = os.environ.get("COMFY_MCP_UPLOAD_TTL_SECONDS", str(_DEFAULT_TTL_SECONDS))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = _DEFAULT_TTL_SECONDS
    if value < _MIN_TTL_SECONDS:
        return _MIN_TTL_SECONDS
    if value > _MAX_TTL_SECONDS:
        return _MAX_TTL_SECONDS
    return value


def bind_host_port() -> tuple[str, int]:
    """Loopback bind for ``comfy-mcp-upload-server``. Port 0 lets tests pick one."""
    host = os.environ.get("COMFY_MCP_UPLOAD_HOST", "127.0.0.1")
    raw_port = os.environ.get("COMFY_MCP_UPLOAD_PORT", "8192")
    if host not in {"127.0.0.1", "::1"}:
        raise ComfyCliError(
            "COMFY_MCP_UPLOAD_HOST must be 127.0.0.1 or ::1. "
            "Publish the upload service through the existing reverse proxy."
        )
    try:
        port = int(raw_port)
    except (TypeError, ValueError):
        port = -1
    if port < 0 or port > 65535:
        raise ComfyCliError("COMFY_MCP_UPLOAD_PORT must be an integer from 0 to 65535.")
    return host, port


def public_base_url() -> str:
    raw = os.environ.get("COMFY_MCP_UPLOAD_PUBLIC_BASE_URL", "").strip().rstrip("/")
    if not raw:
        raise ComfyCliError(
            "COMFY_MCP_UPLOAD_PUBLIC_BASE_URL is not set. Direct upload needs "
            "the public https origin that proxies PUT /upload/ to "
            "comfy-mcp-upload-server."
        )
    if not _ORIGIN_RE.fullmatch(raw):
        raise ComfyCliError(
            "COMFY_MCP_UPLOAD_PUBLIC_BASE_URL must be an https origin with no "
            "userinfo, path, query, or fragment."
        )
    return raw


def payload_path(directory: str, filename: str) -> str:
    """``directory / safe basename``. The client never chooses ``directory``.

    ``filename`` is checked again here so a manifest cannot name a path outside
    the session. The returned path is the file ``comfy upload`` must receive
    once the bytes are complete.
    """
    name = safe_filename(filename)
    if os.path.islink(directory):
        raise ComfyCliError("the staged upload is missing; initialize it again.")
    path = os.path.join(directory, name)
    if os.path.basename(path) != name:
        raise ComfyCliError("the staged upload is missing; initialize it again.")
    try:
        root = os.path.realpath(directory)
        parent = os.path.realpath(os.path.dirname(path))
    except OSError as exc:
        raise ComfyCliError(
            "the staged upload is missing; initialize it again."
        ) from exc
    if parent != root or os.path.islink(path):
        raise ComfyCliError("the staged upload is missing; initialize it again.")
    return path


def safe_filename(name: str, *, fallback: str | None = None) -> str:
    """A basename safe to stage. Never a path the client chose."""
    try:
        return _strict_filename(name)
    except ComfyCliError:
        if fallback is None:
            raise
        return fallback


def check_mime(value: str) -> str:
    if not isinstance(value, str) or not _MIME_RE.fullmatch(value):
        raise ComfyCliError(
            "invalid mime_type: expected a type/subtype such as image/png."
        )
    return value


def initialize(
    filename: str,
    file_size: int,
    mime_type: str,
    overwrite: bool,
) -> dict[str, Any]:
    """Create an ``initialized`` session and return the client payload.

    The payload contains the one-time token. The manifest stores only its hash.
    """
    cleanup_expired()
    filename = safe_filename(filename)
    mime_type = check_mime(mime_type)
    size = _positive_size(file_size)
    base = public_base_url()
    root = _ensure_root()
    upload_id = _mint_id(root)
    directory = os.path.join(root, upload_id)
    os.mkdir(directory, 0o700)
    os.chmod(directory, 0o700)
    token = secrets.token_urlsafe(32)
    now = time.time()
    expires = now + ttl_seconds()
    manifest = {
        "upload_id": upload_id,
        "state": "initialized",
        "filename": filename,
        "safe_filename": filename,
        "mime_type": mime_type,
        "file_size": size,
        "overwrite": bool(overwrite),
        "token_hash": _hash_token(token),
        "created_at_unix": now,
        "expires_at_unix": expires,
        "ready_at_unix": None,
        "sha256": None,
        "byte_size": None,
        "last_error": None,
    }
    try:
        _save(directory, manifest)
    except OSError:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    url = f"{base}/upload/{upload_id}"
    expires_at = datetime.fromtimestamp(expires, timezone.utc).isoformat()
    _event(
        "init_session",
        upload_id,
        "initialized",
        filename=filename,
        mime=mime_type,
        bytes=size,
    )
    return {
        "kind": "upload_initialized",
        "upload_id": upload_id,
        "upload_url": url,
        "upload_headers": {
            "Authorization": f"Bearer {token}",
            "Content-Type": mime_type,
        },
        "expires_at": expires_at,
        "curl_command": _curl(url, token, mime_type),
        "next_tool": "complete_upload",
    }


def accept_put(
    upload_id: str,
    authorization: str | None,
    content_length: str | None,
    content_type: str | None,
    read: Any,
) -> dict[str, int | str]:
    """Stream one PUT to ``payload.part``, then rename it to ``safe_filename``.

    ``read(n)`` returns the next bytes and ``b""`` at EOF. The body is never
    held whole in memory. The session becomes ``ready`` only after that rename.
    """
    try:
        return _accept_put_body(
            upload_id, authorization, content_length, content_type, read
        )
    except UploadRejected as exc:
        if exc.stage == "put":
            exc.stage = _PUT_STAGE.get(exc.status, "put")
        if not exc.upload_id:
            exc.upload_id = upload_id
        _event(exc.stage, exc.upload_id, exc.state, http_status=exc.status)
        raise


def _accept_put_body(
    upload_id: str,
    authorization: str | None,
    content_length: str | None,
    content_type: str | None,
    read: Any,
) -> dict[str, int | str]:
    cleanup_expired()
    directory = _existing_dir(upload_id)
    with _lock(directory):
        manifest = _load(directory)
        _reject_if_expired(directory, manifest)
        if manifest.get("state") == "uploading":
            # The lock is held, so the writer that set ``uploading`` has gone.
            _unlink_quiet(os.path.join(directory, "payload.part"))
            manifest["state"] = "initialized"
            _save(directory, manifest)
        if manifest.get("state") != "initialized":
            raise UploadRejected(
                409,
                "this upload is not accepting another PUT.",
                state=str(manifest.get("state") or ""),
            )
        if not _token_matches(manifest, authorization):
            raise UploadRejected(401, "invalid upload token.", state="initialized")
        length = _declared_length(content_length, manifest)
        _mime_matches(content_type, manifest)
        manifest["state"] = "uploading"
        _save(directory, manifest)
        part = os.path.join(directory, "payload.part")
        try:
            digest, nbytes = _stream_exact(part, read, length, max_upload_bytes())
        except UploadRejected as exc:
            _discard_partial(directory, manifest, exc.message)
            exc.state = "initialized"
            raise
        except OSError:
            _discard_partial(directory, manifest, "upload interrupted")
            raise UploadRejected(
                400,
                "upload interrupted before the declared size.",
                state="initialized",
            ) from None
        except Exception as exc:  # noqa: BLE001 - a reset must not leave the session ready
            _discard_partial(directory, manifest, "upload interrupted")
            _LOG.exception(
                "upload_id=%s stage=put_body state=uploading error_type=%s",
                upload_id,
                type(exc).__name__,
            )
            raise UploadRejected(
                400,
                "upload interrupted before the declared size.",
                state="initialized",
            ) from None
        try:
            ready = payload_path(directory, _manifest_filename(manifest))
            if part != ready:
                os.replace(part, ready)
            os.chmod(ready, 0o600)
        except (ComfyCliError, OSError):
            _discard_partial(directory, manifest, "upload could not be stored")
            raise UploadRejected(500, "upload could not be stored.") from None
        manifest["state"] = "ready"
        manifest["token_hash"] = ""
        manifest["sha256"] = digest
        manifest["byte_size"] = nbytes
        manifest["ready_at_unix"] = time.time()
        manifest["last_error"] = None
        _save(directory, manifest)
        _event(
            "put_commit",
            upload_id,
            "ready",
            filename=manifest.get("safe_filename"),
            bytes=nbytes,
            sha256=digest,
        )
        return {"byte_size": nbytes, "sha256": digest}


def begin_complete(upload_id: str) -> ReadyUpload:
    """Move ``ready`` → ``completing`` and return the staged file."""
    _event("load_session", upload_id)
    try:
        cleanup_expired()
        directory = _existing_dir(upload_id, tool=True)
    except ComfyCliError:
        raise UploadFailed(
            "load_session",
            "complete_upload failed: unknown upload session.",
            upload_id=upload_id if isinstance(upload_id, str) else None,
            retryable=False,
        ) from None
    with _lock(directory):
        try:
            manifest = _load(directory, tool=True)
        except ComfyCliError:
            raise UploadFailed(
                "load_session",
                "complete_upload failed: the upload session is unreadable.",
                upload_id=upload_id,
                retryable=False,
            ) from None
        state = manifest.get("state")
        state_text = state if isinstance(state, str) else None
        _event("validate_state", upload_id, state_text)
        try:
            _reject_if_expired(directory, manifest, tool=True)
        except ComfyCliError:
            raise UploadFailed(
                "validate_state",
                "complete_upload failed: the upload session has expired.",
                upload_id=upload_id,
                state=state_text,
                retryable=False,
            ) from None
        if state != "ready":
            raise UploadFailed(
                "validate_state",
                "complete_upload failed: " + _complete_state_message(state),
                upload_id=upload_id,
                state=state_text,
                retryable=state in {"initialized", "uploading"},
            )
        try:
            filename = _manifest_filename(manifest)
        except ComfyCliError:
            raise UploadFailed(
                "validate_state",
                "complete_upload failed: the upload session is incomplete.",
                upload_id=upload_id,
                state=state_text,
                retryable=False,
            ) from None
        _event("resolve_staged_file", upload_id, "ready", filename=filename)
        try:
            ready = payload_path(directory, filename)
        except ComfyCliError:
            raise UploadFailed(
                "resolve_staged_file",
                "complete_upload failed: the staged filename is not usable.",
                upload_id=upload_id,
                state="ready",
                retryable=False,
            ) from None
        if not os.path.isfile(ready) or os.path.islink(ready):
            entries = _session_entry_names(directory)
            _LOG.info(
                "upload_id=%s stage=resolve_staged_file state=ready "
                "ready upload has no staged safe_filename; session entries=%s",
                upload_id,
                entries,
            )
            shown = filename.replace('"', "'")
            raise UploadFailed(
                "resolve_staged_file",
                "complete_upload failed: "
                f'staged file "{shown}" is missing for a ready upload session.',
                upload_id=upload_id,
                state="ready",
                retryable=False,
            )
        source = manifest.get("filename")
        if not isinstance(source, str) or not source:
            source = filename
        mime = manifest.get("mime_type")
        sha = manifest.get("sha256")
        size = manifest.get("byte_size")
        if (
            not isinstance(mime, str)
            or not isinstance(sha, str)
            or not isinstance(size, int)
        ):
            raise UploadFailed(
                "validate_state",
                "complete_upload failed: the upload session is incomplete.",
                upload_id=upload_id,
                state="ready",
                retryable=False,
            )
        if os.path.basename(ready) != filename:
            raise UploadFailed(
                "resolve_staged_file",
                "complete_upload failed: the staged filename is not usable.",
                upload_id=upload_id,
                state="ready",
                retryable=False,
            )
        manifest["state"] = "completing"
        manifest["expires_at_unix"] = time.time() + ttl_seconds()
        _save(directory, manifest)
        _event("lock_completion", upload_id, "completing", filename=source)
        return ReadyUpload(
            upload_id=upload_id,
            path=ready,
            filename=source,
            mime_type=mime,
            byte_size=size,
            sha256=sha,
            overwrite=bool(manifest.get("overwrite")),
        )


def abort_complete(upload_id: str, reason: str) -> None:
    """Return a ``completing`` session to ``ready`` and keep the staged file."""
    try:
        directory = _existing_dir(upload_id, tool=True)
    except ComfyCliError:
        return
    note = (
        reason
        if reason and "\n" not in reason and len(reason) <= 200
        else "comfy upload failed"
    )
    with _lock(directory):
        if not os.path.isdir(directory):
            return
        manifest = _load(directory)
        if manifest.get("state") != "completing":
            return
        manifest["state"] = "ready"
        manifest["last_error"] = note
        _save(directory, manifest)
        _event("abort_complete", upload_id, "ready")


def finish_complete(upload_id: str) -> None:
    """Delete a session after ``comfy upload`` has accepted the file."""
    root = _ensure_root()
    directory = os.path.join(root, upload_id)
    if not _inside(root, directory) or not os.path.isdir(directory):
        return
    with _lock(directory):
        shutil.rmtree(directory, ignore_errors=True)
    _event("cleanup", upload_id, "completed")


def cleanup_expired(now: float | None = None) -> None:
    """Remove expired session directories under the spool and nothing else."""
    root = spool_dir()
    if not os.path.isdir(root) or os.path.islink(root):
        return
    moment = time.time() if now is None else now
    try:
        names = os.listdir(root)
    except OSError:
        return
    for name in names:
        if not _ID_RE.fullmatch(name):
            continue
        directory = os.path.join(root, name)
        if os.path.islink(directory) or not os.path.isdir(directory):
            continue
        try:
            with _lock(directory):
                if not os.path.isdir(directory):
                    continue
                manifest = _read(directory)
                if manifest is None:
                    if moment - os.path.getmtime(directory) >= ttl_seconds():
                        shutil.rmtree(directory, ignore_errors=True)
                    continue
                expires = manifest.get("expires_at_unix")
                if not isinstance(expires, (int, float)) or expires > moment:
                    continue
                shutil.rmtree(directory, ignore_errors=True)
                _event(
                    "expire",
                    name,
                    manifest.get("state")
                    if isinstance(manifest.get("state"), str)
                    else None,
                )
        except OSError:
            continue


def _curl(url: str, token: str, mime: str) -> str:
    return (
        "curl --fail-with-body \\\n"
        "  --request PUT \\\n"
        '  --upload-file "$FILE" \\\n'
        f'  -H "Authorization: Bearer {token}" \\\n'
        f'  -H "Content-Type: {mime}" \\\n'
        f'  "{url}"'
    )


def _positive_size(file_size: object) -> int:
    if type(file_size) is not int or isinstance(file_size, bool):
        raise ComfyCliError(
            "invalid file_size: expected a positive integer of original bytes."
        )
    if file_size < 1:
        raise ComfyCliError(
            "invalid file_size: expected a positive integer of original bytes."
        )
    if file_size > max_upload_bytes():
        raise ComfyCliError(
            "file_size exceeds COMFY_MCP_MAX_UPLOAD_MB "
            f"({max_upload_bytes() // (1024 * 1024)} MB of original bytes)."
        )
    return file_size


def _strict_filename(name: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name.strip() != name
        or name.strip() == ""
    ):
        raise ComfyCliError("invalid filename: pass a basename, not a path.")
    if (
        name in {".", ".."}
        or any(ord(c) < 32 for c in name)
        or any(c in name for c in "/\\:\x00")
    ):
        raise ComfyCliError("invalid filename: pass a basename, not a path.")
    if _DEVICE_RE.match(name):
        raise ComfyCliError("invalid filename: pass a basename, not a path.")
    if len(name.encode("utf-8")) > _MAX_FILENAME_BYTES:
        raise ComfyCliError("invalid filename: longer than 255 bytes.")
    return name


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _bearer(authorization: str | None) -> str:
    if not isinstance(authorization, str):
        return ""
    prefix = "Bearer "
    if (
        len(authorization) <= len(prefix)
        or authorization[: len(prefix)].lower() != prefix.lower()
    ):
        return ""
    token = authorization[len(prefix) :]
    if not token or any(c.isspace() for c in token):
        return ""
    return token


def _token_matches(manifest: dict[str, Any], authorization: str | None) -> bool:
    stored = manifest.get("token_hash")
    if not isinstance(stored, str) or len(stored) != 64:
        return False
    digest = _hash_token(_bearer(authorization))
    return hmac.compare_digest(digest, stored)


def _declared_length(content_length: str | None, manifest: dict[str, Any]) -> int:
    if content_length is None or content_length == "":
        raise UploadRejected(411, "Content-Length is required.")
    try:
        length = int(content_length)
    except (TypeError, ValueError):
        raise UploadRejected(400, "Content-Length is not an integer.") from None
    if length < 0:
        raise UploadRejected(400, "Content-Length is not an integer.")
    limit = max_upload_bytes()
    if length > limit:
        raise UploadRejected(413, "upload exceeds COMFY_MCP_MAX_UPLOAD_MB.")
    expected = manifest.get("file_size")
    if length != expected:
        raise UploadRejected(
            400, "Content-Length does not match the declared file_size."
        )
    return length


def _mime_matches(content_type: str | None, manifest: dict[str, Any]) -> None:
    if not isinstance(content_type, str) or not content_type.strip():
        raise UploadRejected(415, "Content-Type does not match the declared mime_type.")
    got = content_type.split(";", 1)[0].strip()
    if got != manifest.get("mime_type"):
        raise UploadRejected(415, "Content-Type does not match the declared mime_type.")


def _stream_exact(path: str, read: Any, expected: int, limit: int) -> tuple[str, int]:
    hasher = hashlib.sha256()
    written = 0
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.chmod(path, 0o600)
        while written < expected:
            want = min(_CHUNK, expected - written)
            chunk = read(want)
            if not isinstance(chunk, (bytes, bytearray)) or not chunk:
                raise UploadRejected(400, "upload ended before the declared file_size.")
            if len(chunk) > want:
                raise UploadRejected(413, "upload exceeds COMFY_MCP_MAX_UPLOAD_MB.")
            written += len(chunk)
            if written > limit or written > expected:
                raise UploadRejected(413, "upload exceeds COMFY_MCP_MAX_UPLOAD_MB.")
            hasher.update(chunk)
            view = memoryview(chunk)
            while view:
                n = os.write(fd, view)
                view = view[n:]
        os.fsync(fd)
    except Exception:
        os.close(fd)
        raise
    else:
        os.close(fd)
    return hasher.hexdigest(), written


def _discard_partial(directory: str, manifest: dict[str, Any], reason: str) -> None:
    part = os.path.join(directory, "payload.part")
    try:
        os.unlink(part)
    except FileNotFoundError:
        pass
    manifest["state"] = "initialized"
    manifest["last_error"] = reason
    manifest["sha256"] = None
    manifest["byte_size"] = None
    _save(directory, manifest)


def _session_entry_names(directory: str) -> list[str]:
    """Basenames in one session directory. Never a path outside it."""
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    return sorted(name for name in names if name not in {".", ".."})


def _manifest_filename(manifest: dict[str, Any]) -> str:
    """Sanitized basename stored by ``init_upload``. Never a client path."""
    name = manifest.get("safe_filename")
    if not isinstance(name, str) or not name:
        name = manifest.get("filename")
    if not isinstance(name, str) or not name:
        raise ComfyCliError("the upload session is incomplete; initialize it again.")
    return name


def _complete_state_message(state: object) -> str:
    if state == "initialized":
        return "the upload has not received its bytes yet; PUT the file, then call complete_upload."
    if state == "uploading":
        return "the upload is still transferring; call complete_upload after the PUT finishes."
    if state == "completing":
        return "this upload is already being completed."
    if state == "completed":
        return "this upload was already completed."
    return "this upload cannot be completed; initialize it again."


def _reject_if_expired(
    directory: str, manifest: dict[str, Any], *, tool: bool = False
) -> None:
    expires = manifest.get("expires_at_unix")
    if isinstance(expires, (int, float)) and expires <= time.time():
        shutil.rmtree(directory, ignore_errors=True)
        if tool:
            raise ComfyCliError("the upload session has expired; initialize it again.")
        raise UploadRejected(401, "the upload session has expired.")


def _existing_dir(upload_id: str, *, tool: bool = False) -> str:
    if not isinstance(upload_id, str) or not _ID_RE.fullmatch(upload_id):
        if tool:
            raise ComfyCliError("unknown upload_id.")
        raise UploadRejected(404, "unknown upload.")
    root = spool_dir()
    directory = os.path.join(root, upload_id)
    if (
        not _inside(root, directory)
        or os.path.islink(directory)
        or not os.path.isdir(directory)
    ):
        if tool:
            raise ComfyCliError("unknown upload_id.")
        raise UploadRejected(404, "unknown upload.")
    return directory


def _inside(root: str, directory: str) -> bool:
    try:
        root_real = os.path.realpath(root)
        directory_real = os.path.realpath(directory)
    except OSError:
        return False
    if not os.path.isdir(root_real):
        return False
    try:
        return os.path.commonpath([root_real, directory_real]) == root_real
    except ValueError:
        return False


def _ensure_root() -> str:
    root = spool_dir()
    try:
        os.makedirs(root, mode=0o700, exist_ok=True)
        if os.path.islink(root):
            raise ComfyCliError("COMFY_MCP_UPLOAD_DIR must be a real directory.")
        os.chmod(root, 0o700)
    except OSError as exc:
        raise ComfyCliError(
            "cannot create the upload spool (COMFY_MCP_UPLOAD_DIR)."
        ) from exc
    return root


def _mint_id(root: str) -> str:
    for _ in range(8):
        upload_id = secrets.token_urlsafe(18)
        if not _ID_RE.fullmatch(upload_id):
            continue
        if not os.path.exists(os.path.join(root, upload_id)):
            return upload_id
    raise ComfyCliError("could not allocate an upload id.")


def _save(directory: str, manifest: dict[str, Any]) -> None:
    blob = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    tmp = os.path.join(directory, "manifest.json.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, blob)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(tmp, 0o600)
    os.replace(tmp, os.path.join(directory, "manifest.json"))


def _load(directory: str, *, tool: bool = False) -> dict[str, Any]:
    manifest = _read(directory)
    if manifest is None:
        if tool:
            raise ComfyCliError(
                "the upload session is unreadable; initialize it again."
            )
        raise UploadRejected(500, "upload session is unreadable.")
    return manifest


def _unlink_quiet(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _read(directory: str) -> dict[str, Any] | None:
    path = os.path.join(directory, "manifest.json")
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


@contextmanager
def _lock(directory: str) -> Iterator[None]:
    fd = os.open(os.path.join(directory, ".lock"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.chmod(os.path.join(directory, ".lock"), 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
