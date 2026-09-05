"""Remote-target resolution, redaction, and provenance for the run/job tools.

Leaf module over :mod:`comfy_mcp.errors` and :mod:`comfy_mcp.textutil`: it owns
everything about pointing the run/job tools (``run_workflow``, ``run_template``,
``generate_image``, the ``jobs``/``upload`` verbs) at a ComfyUI running
ELSEWHERE via ``COMFYUI_URL`` / ``COMFYUI_HOST`` (+ optional ``COMFYUI_PORT``)
instead of the implicit local default — resolution (:func:`_comfy_target`),
the ``--host``/``--port`` forward (:func:`_with_target`), userinfo/credential
redaction on the way back to the client (:func:`_redact_config_url`,
:func:`_redact_target_host`), the local-only ``download_model`` refusal
(:func:`_reject_remote_model_download`), and the divergence notes two
LOCAL-only tools (``system_stats`` / ``free_memory``) attach so an agent does
not gate a remote run on local numbers (:func:`_annotate_comfy_target`,
:func:`_target_provenance_suffix`, :func:`_with_target_provenance`). Nothing
here imports ``server``.

Everything here is private and reached as ``target._name`` — there is no
public name in this module.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

from . import textutil
from .errors import ComfyCliError

# Optional: point the run/queue tools at a ComfyUI running ELSEWHERE (e.g. a GPU
# box reachable over a private network / tailnet) instead of the implicit local
# 127.0.0.1:8188. Configure with a single ``COMFYUI_URL`` (e.g.
# ``http://10.0.0.5:8188``) OR the ``COMFYUI_HOST`` (+ optional ``COMFYUI_PORT``)
# pair. When UNSET the tools behave byte-identically to the local-only default
# (no ``--host`` forwarded); when set, ``_with_target`` forwards ``--host`` /
# ``--port`` to the comfy-cli verbs that accept them (see ``_comfy_target`` /
# ``_with_target`` and ``_TARGET_AWARE_SUBCOMMANDS`` below).
DEFAULT_COMFYUI_PORT = 8188

# Escape hatch for the one tool that REFUSES to run against a configured remote:
# ``download_model``. ``comfy model download`` has no target concept at all — it
# writes into the workspace of the machine running this MCP server — so with a
# remote configured it would land the checkpoint on the wrong disk and the run
# that needed it would still fail on a missing model. That combination is
# refused (see ``_reject_remote_model_download``).
#
# The one deployment where today's behavior is CORRECT is shared storage: an NFS
# / tailnet mount where this machine's workspace models dir IS the remote's. No
# environment check can tell that apart from the wrong-disk case, and this
# server may not probe the remote to find out (AGENTS.md: comfy-cli owns all
# I/O), so it is an explicit operator assertion rather than a detection. Set it
# to ``1`` to skip the guard.
#
# Read at CALL time, never cached, so it tracks the environment the same way
# ``COMFYUI_URL`` / ``COMFYUI_HOST`` do. Only the spellings below count: an
# unrecognized value leaves the guard ARMED (fail closed), and the guard's own
# message names the value to set.
REMOTE_SHARED_MODELS_ENV = "COMFY_MCP_REMOTE_SHARED_MODELS"
_REMOTE_SHARED_MODELS_TRUE = frozenset({"1", "true", "yes", "on"})

# The comfy-cli verbs this server forwards ``--host`` / ``--port`` to: ``comfy
# run``, ``comfy run-template``, every ``comfy jobs`` subcommand, and ``comfy
# upload`` — every verb that SUBMITS a job, reads one back, or stages the files
# a job will read, which is the set that has to agree on which server it is
# talking to. ``run`` / ``jobs`` / ``upload`` are the set comfy-cli's
# ``comfy_cli/host_port.py`` documents; ``run-template`` declares the same two
# options and resolves them through that module's own ``resolve_host_port``
# (``comfy_cli/command/templates.py``), it is just missing from that docstring's
# list.
#
# ``upload`` is here because an input file is only useful on the machine that
# RUNS the workflow reading it. Without the forward, a session with a remote
# configured staged ``upload_file``'s images into the LOCAL install's ``input``
# directory while ``run_workflow`` submitted to the remote, which then failed on
# a filename it could not see — the same submit/poll-must-agree argument as
# ``run-template``, one step earlier in the flow. comfy-cli resolves the pair for
# this verb through ``target.resolve_target`` rather than ``resolve_host_port``
# (no persisted-background-server fallback), which changes nothing here: this
# server either forwards an explicit host/port or forwards neither.
#
# Unlike ``run-template``, this one gets an explicit VERSION-SKEW error rather
# than relying on the verb's own age. ``comfy upload`` long predates its
# ``--host`` / ``--port`` options — they landed in comfy-cli 1.14.0, which is
# exactly ``_MIN_COMFY_CLI`` — so an older CLI has the verb and rejects only the
# flags, and ``_check_comfy_version`` fails OPEN (an unparseable / erroring /
# timing-out ``--version`` lets a stale install through). ``upload_file``
# therefore translates Click's "No such option" into an upgrade instruction
# instead of leaving a raw usage dump, and deliberately does NOT retry without
# the flags: the caller configured a remote, and a silent local upload is the
# exact wrong-machine bug this forward fixes. It is not a dead end either — that
# error names the workaround that does work on such a CLI, since comfy-cli
# 1.13.0's ``upload`` already resolved its address through
# ``local_address.resolve_local_host_port``, which honors ``COMFY_LOCAL_URL``
# for ANY host (it is not loopback-restricted).
#
# ``run-template`` is here because SUBMIT and POLL must land on the SAME server.
# It backs both ``run_template`` and ``generate_image``, and while it was absent
# the submit-then-poll flow did not merely run on the wrong box, it could not
# complete at all: the run executed locally, ``job(action="wait")`` (a ``jobs``
# verb, forwarded) polled the configured remote, and the local ``prompt_id`` came back
# ``prompt_not_found`` from a queue it was never submitted to. Adding a verb here
# is only safe when comfy-cli actually accepts the flags on it — otherwise the
# forward turns every call into "No such option". There is no such window for
# this one: ``run-template`` shipped WITH both options, in the same comfy-cli
# release (1.13.0) that introduced the verb — which is at or below the floor
# ``_check_comfy_version`` enforces (``_MIN_COMFY_CLI``). A comfy-cli old enough
# to reject ``--host`` here has no ``run-template`` at all, so it fails on the
# verb before the flags are ever parsed. That argument is about RELEASED
# comfy-cli, and the floor guard deliberately fails OPEN, so a fork or source
# build carrying ``run-template`` with its options stripped would still get a
# Click usage error rather than a graceful degrade. No probe is added for it
# because the exposure is not this verb's: ``run`` and ``jobs`` have been
# forwarded unprobed all along and would fail the same way. Probing (the
# ``_comfy_run_takes_allow_spend`` shape) is the fix if that ever stops being
# hypothetical — for ALL THREE verbs, not just this one.
#
# Deliberately NOT forwarded:
#   * ``env`` / ``download`` / ``templates`` / ``models`` /
#     ``generate`` / the lifecycle verbs take NO ``--host`` / ``--port`` at all,
#     so forwarding would error "No such option" — they stay local-only. That is
#     not always a functional limit: ``download`` still collects a REMOTE job's
#     files, because it resolves the ``prompt_id`` from the state file this
#     machine wrote at submit (which records absolute ``/view`` URLs on the
#     remote) instead of asking a server — see ``fetch_outputs``.
#   * ``nodes`` / ``validate`` DO accept ``--host`` / ``--port`` in current
#     comfy-cli, but remoting live discovery/validation is out of scope here;
#     forwarding them is a clean follow-up. Their local answers are advisory —
#     ``run_workflow`` and ``run_template`` already submit to a remote whose node
#     set a local check cannot see.
# Forwarding is a no-op for the local default regardless, so unconfigured
# behavior is unchanged for every tool.
_TARGET_AWARE_SUBCOMMANDS = frozenset({"run", "run-template", "jobs", "upload"})


def _strip_brackets(host: str) -> str:
    """Strip surrounding ``[...]`` from a bracketed IPv6 host for consistency.

    ``urlparse`` already returns an IPv6 ``.hostname`` bracket-free, so normalize
    a bracketed ``COMFYUI_HOST`` (``[::1]``) the same way — both config paths then
    forward a bare host to comfy-cli, which re-brackets it when building its URL.
    """
    if host.startswith("[") and host.endswith("]"):
        return host[1:-1]
    return host


def _redact_config_url(url: str) -> str:
    """:func:`textutil._redact_url`, plus the query/fragment/params a secret hides in.

    :func:`_comfy_target`'s errors echo the offending ``COMFYUI_URL`` back, and
    that text is quoted onward — into ``server_info``'s ``comfy_target`` block,
    into :func:`_malformed_target_note`, and into the message
    :func:`_target_provenance_suffix` appends to a raised error — so it reaches
    the model's context and any transcript of it. Userinfo is not the only place
    a credential rides in: an auth-proxied ComfyUI is routinely addressed as
    ``https://host/?token=…``, and a value written that way is echoed back the
    moment ANY of the checks below rejects it — the ``https`` scheme most of
    all, which is what such a proxy is usually reached over.

    ``;`` is cut alongside them because ``urlparse`` splits ``;params`` off the
    last path segment: ``http://host:8188/;token=…`` carries a secret with no
    ``?`` anywhere in it, and is rejected (and echoed) by the same check. It is
    searched only from after the scheme separator, the way
    :func:`textutil._redact_url` bounds its own netloc scan — a ``;`` before that
    is inside the scheme, and cutting there would take the host and port with it.

    All three are replaced WHOLESALE rather than parsed for key names: what makes
    the value diagnosable is the scheme, host and port, so nothing in any of them
    is worth echoing — and a placeholder still tells the user they wrote one,
    which "silently dropped" would not.
    """
    masked = textutil._redact_url(url)
    scheme_sep = masked.find("://")
    semi_from = scheme_sep + 3 if scheme_sep != -1 else 0
    cuts = [
        i
        for i in (masked.find("?"), masked.find("#"), masked.find(";", semi_from))
        if i != -1
    ]
    if not cuts:
        return masked
    cut = min(cuts)
    return f"{masked[: cut + 1]}<redacted>"


def _comfy_target() -> tuple[str, int, str] | None:
    """Resolve the configured ComfyUI ``(host, port, source)``, or None for local.

    Precedence: ``COMFYUI_URL`` (a full URL, parsed into host + port) wins;
    otherwise ``COMFYUI_HOST`` (+ optional ``COMFYUI_PORT``, default
    :data:`DEFAULT_COMFYUI_PORT`). Returns ``None`` when nothing is set, so the
    tools stay byte-identical to the local-only default (no ``--host`` forwarded,
    comfy-cli's own 127.0.0.1:8188). Raises :class:`ComfyCliError` on a set but
    malformed value rather than silently retargeting to the wrong place.

    comfy-cli's ``--host`` / ``--port`` carry only a host and port, so a
    ``COMFYUI_URL`` that also names a non-``http`` scheme (``https://``), a base
    path (``/comfyui``), or a query / fragment / ``;params`` (``?token=…``) is
    REJECTED rather than silently dropped — otherwise a user asking for TLS, a
    reverse-proxy path, or an auth proxy's token would be quietly downgraded to a
    plain unauthenticated request against the bare host.
    """
    url = os.environ.get("COMFYUI_URL", "").strip()
    if url:
        # urlparse needs a scheme to populate .hostname/.port; a bare
        # "host:port" is otherwise read as scheme:path. Prefix "//" so a
        # scheme-less value parses as a netloc. urlparse itself raises
        # ValueError on a malformed value (e.g. an unbalanced IPv6 bracket
        # "http://[::1"), so it lives inside the try alongside the .port access.
        try:
            parsed = urlparse(url if "://" in url else f"//{url}")
            host, port = parsed.hostname, parsed.port
        except ValueError as exc:  # bad port, or malformed URL (IPv6 brackets)
            raise ComfyCliError(
                f"COMFYUI_URL is malformed: {_redact_config_url(url)!r} ({exc})."
            ) from exc
        if parsed.scheme and parsed.scheme != "http":
            raise ComfyCliError(
                f"COMFYUI_URL scheme {parsed.scheme!r} is not supported "
                f"({_redact_config_url(url)!r}): comfy-cli's --host/--port speak plain "
                "http only, so an https:// target would be silently downgraded. "
                "Use http://<host>:<port>."
            )
        if parsed.path not in ("", "/"):
            raise ComfyCliError(
                f"COMFYUI_URL must not include a path ({_redact_config_url(url)!r}): "
                "comfy-cli forwards only host/port, so a reverse-proxy base path "
                "would be dropped. Point COMFYUI_URL at the bare host:port."
            )
        # `params` is the `;token=abc` spelling: `urlparse` splits it off the
        # last path segment, so `http://host:8188/;token=abc` leaves `path` at
        # "/" and slips past the check above. Same silent drop, same rejection.
        if parsed.query or parsed.fragment or parsed.params:
            raise ComfyCliError(
                f"COMFYUI_URL must not include a query or fragment "
                f"({_redact_config_url(url)!r}): comfy-cli forwards only "
                "host/port, so a query or fragment (e.g. a proxy auth "
                "?token=...) would be silently dropped. The ';token=...' "
                "spelling goes the same way — it is a path parameter, so it is "
                "neither the path this checks above nor a query. Point "
                "COMFYUI_URL at the bare host:port and carry the credential "
                "another way."
            )
        if not host:
            raise ComfyCliError(
                f"COMFYUI_URL is set but names no host: {_redact_config_url(url)!r}. "
                "Use e.g. http://<host>:8188 (or set COMFYUI_HOST/COMFYUI_PORT)."
            )
        # `port or DEFAULT` alone would treat an explicit :0 as absent and
        # silently target 8188; reject it to match the COMFYUI_PORT path.
        if port == 0:
            raise ComfyCliError(
                f"COMFYUI_URL port is out of range (1-65535): {_redact_config_url(url)!r}."
            )
        return _strip_brackets(host), port or DEFAULT_COMFYUI_PORT, "COMFYUI_URL"

    host = os.environ.get("COMFYUI_HOST", "").strip()
    raw_port = os.environ.get("COMFYUI_PORT", "").strip()
    if not host:
        # A port alone does not select a remote; raise rather than silently
        # ignoring it and defaulting back to the local 127.0.0.1:8188.
        if raw_port:
            raise ComfyCliError(
                "COMFYUI_PORT is set but COMFYUI_HOST is not; a port alone does "
                "not select a remote. Set COMFYUI_HOST (or COMFYUI_URL) to "
                "target a remote ComfyUI."
            )
        return None
    if not raw_port:
        return _strip_brackets(host), DEFAULT_COMFYUI_PORT, "COMFYUI_HOST"
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ComfyCliError(
            f"COMFYUI_PORT must be an integer, got {raw_port!r}."
        ) from exc
    if not (1 <= port <= 65535):
        raise ComfyCliError(f"COMFYUI_PORT is out of range (1-65535): {port}.")
    return _strip_brackets(host), port, "COMFYUI_HOST"


def _redact_target_host(host: str) -> str:
    """Mask any ``user:pass@`` before a resolved target host is echoed back.

    ``COMFYUI_URL`` is parsed with :func:`urlparse`, whose ``.hostname`` already
    drops userinfo — but ``COMFYUI_HOST`` is taken VERBATIM, so a value written
    URL-style (``<user>:<token>@host``) is carried into the tuple as-is and would
    otherwise reach the model's context, and any transcript of it, unmasked. The
    same userinfo masking :func:`_comfy_target`'s own error messages apply to the
    raw value; a host with no ``@`` is returned unchanged, which is every real
    one.

    Masked HERE rather than by delegating to :func:`textutil._redact_url`: that
    helper stops inspecting at the first ``/``, ``?`` or ``#`` because it is
    reading a URL, whose PATH may hold a stray ``@`` that is not userinfo. This
    is a bare host, where there is no path for an ``@`` to belong to — so any
    ``@`` is userinfo, and a token that happens to contain one of those
    delimiters (base64 routinely contains ``/``) would otherwise fall outside
    the inspected slice and be returned in full.
    """
    if "@" not in host:
        return host
    return f"***@{host.rsplit('@', 1)[1]}"


def _format_target_endpoint(host: str, port: int) -> str:
    """``host:port`` as ONE token, re-bracketing an IPv6 host so the port reads.

    :func:`_strip_brackets` normalizes ``[2001:db8::1]`` to the bare form every
    other consumer wants (comfy-cli re-brackets it when it builds a URL, and the
    payload note keeps ``host`` and ``port`` as separate keys). Joined with a
    colon for prose, though, that bare form renders ``2001:db8::1:8188``, where
    the port is indistinguishable from a final hextet — so the brackets go back
    on for the one caller that has to write the two as a single string.
    """
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def _malformed_target_note(exc: ComfyCliError) -> dict[str, str]:
    """The error-shaped remote-target block for a malformed ``COMFYUI_URL``/``_HOST``.

    Shared by ``server_info`` and :func:`_annotate_comfy_target` so a broken
    config reads the same wherever it surfaces: a diagnostic FIELD rather than a
    raise, since the tools carrying it either never touch the remote
    (``system_stats`` / ``free_memory``) or exist to explain the environment
    (``server_info``). What it must not do is look like "nothing configured" —
    no remote resolves either way, but only one of the two is a typo the user
    wants to hear about. The message is :func:`_comfy_target`'s, already
    userinfo-masked.
    """
    return {
        "error": str(exc),
        "note": (
            "COMFYUI_URL/COMFYUI_HOST is set but malformed, so no remote is "
            "resolved; the submit/poll tools (run_workflow, generate_image, "
            "run_template and the jobs/queue tools) will raise this same error "
            "until it is fixed."
        ),
    }


def _with_target(args: tuple[str, ...]) -> tuple[str, ...]:
    """Append ``--host`` / ``--port`` to a target-aware subcommand, if configured.

    The flags are injected into the SUBCOMMAND args (after the ``run`` /
    ``run-template`` / ``jobs`` / ``upload`` verb), never into the global
    ``--json`` / ``--where`` prefix, since ``--host`` / ``--port`` are
    subcommand options. A no-op for the local default (``_comfy_target`` is
    None) and for any subcommand that doesn't accept the flags (see
    :data:`_TARGET_AWARE_SUBCOMMANDS`), so unconfigured behavior is
    byte-identical to today.

    They go at the END of the subcommand args, past any positionals — Click
    parses options wherever they appear, including after ``upload``'s one
    guarded file-path positional.
    """
    # Check the verb FIRST, then resolve the target. A malformed
    # COMFYUI_URL/PORT must not brick local-only verbs (server_info's `env`,
    # download, the stop/logs lifecycle) that never touch the remote — they'd
    # otherwise raise ComfyCliError here despite ignoring the target entirely,
    # breaking the "local behavior unchanged" contract.
    if not args or args[0] not in _TARGET_AWARE_SUBCOMMANDS:
        return args
    target = _comfy_target()
    if target is None:
        return args
    host, port, _source = target
    return (*args, "--host", host, "--port", str(port))


def _remote_shared_models_optin() -> bool:
    """True when the operator asserted this workspace IS the remote's models dir.

    See :data:`REMOTE_SHARED_MODELS_ENV`. Fails CLOSED: anything outside the
    recognized truthy spellings — including an empty value, ``0`` and ``false``
    — leaves :func:`_reject_remote_model_download`'s guard armed.
    """
    return (
        os.environ.get(REMOTE_SHARED_MODELS_ENV, "").strip().lower()
        in _REMOTE_SHARED_MODELS_TRUE
    )


def _reject_remote_model_download() -> None:
    """Refuse a model download while a REMOTE ComfyUI target is configured.

    ``comfy model download`` is not in :data:`_TARGET_AWARE_SUBCOMMANDS` and has
    no ``--host`` / ``--port`` to be added to it: comfy-cli resolves the
    destination from ``get_workspace()``, i.e. the models directory of the
    machine running THIS server. So with ``COMFYUI_URL`` / ``COMFYUI_HOST``
    pointing elsewhere, a download "succeeds" onto the wrong disk — the remote
    never sees the file, and the ``run_workflow`` that needed it still fails on a
    missing model. Refuse early instead: a clear failure beats a silent success
    on the wrong machine.

    There is no remote-download mode to fall back to. Nothing in comfy-cli
    accepts a target for this verb, and ComfyUI's own HTTP surface has no
    endpoint that writes into the models directory (its uploads land in
    ``input``), so this cannot be implemented here without an upstream change —
    which is also why the guard names the ways AROUND it rather than a flag that
    would make it work.

    Deliberately NOT folded into :func:`_with_target`: that helper checks the
    verb before resolving the target precisely so a malformed ``COMFYUI_URL``
    cannot brick local-only verbs. Here the opposite is wanted — the
    caller has opted into a remote, so a malformed value should fail loudly
    rather than be ignored, and the raise comes straight out of
    :func:`_comfy_target`.
    """
    if _remote_shared_models_optin():
        return
    target = _comfy_target()
    if target is None:
        return
    host, port, source = target
    raise ComfyCliError(
        "download_model is LOCAL-ONLY, but a remote ComfyUI is configured "
        f"({source} -> {host}:{port}). `comfy model download` takes no "
        "--host/--port: it writes into the models directory of the machine "
        "running this MCP server, so that remote would never see the file and a "
        "run there would still fail on a missing model. Either (1) run the "
        "download on the remote host itself (its own comfy-cli / MCP server), "
        "or (2) if this machine's workspace models directory IS the remote's "
        "(shared storage — an NFS / tailnet mount), set "
        f"{REMOTE_SHARED_MODELS_ENV}=1 to allow the download, or (3) unset "
        "COMFYUI_URL/COMFYUI_HOST to work entirely locally. Already-submitted "
        "downloads are unaffected: download(action='status'/'wait'/'cancel') "
        "keeps working."
    )


_COMFY_TARGET_NOTE_KEY = "comfy_target_note"


def _annotate_comfy_target(payload: Any) -> Any:
    """Return *payload* tagged with the configured remote, for a LOCAL-only tool.

    ``comfy system-stats`` and ``comfy free`` take no ``--host`` / ``--port``
    (only ``--where``), so they read and act on whichever ComfyUI comfy-cli
    itself targets — while ``run_workflow`` submits to the ``COMFYUI_URL`` /
    ``COMFYUI_HOST`` remote. Both docstrings say so, but an agent that skipped
    them sees numbers with no provenance and gates a remote run on local VRAM.
    So the divergence is carried IN the payload: a ``comfy_target_note`` key
    naming the configured target and stating that this payload need not be
    about it.

    A WARNING, deliberately not an error: freeing local VRAM while submitting a
    remote run is a legitimate pattern (the local-LLM coexistence recipe in the
    README keeps it), so the call still succeeds and the payload still carries
    everything comfy-cli returned.

    The note reports the divergence, it does not ADJUDICATE it — the two ends
    are the same machine whenever the configured host resolves to this box, and
    this server cannot tell whether it does. It carries no local hostname or
    interface addresses, a loopback host can be an SSH tunnel to a remote GPU,
    and ``COMFY_LOCAL_URL`` can repoint comfy-cli off-box with no target
    configured here at all — the same reason the routing rule at the top of this
    module tells the agent to ASK about a host it cannot place. So the note
    hands over the host/port and says what is and is not verified, exactly as
    ``server_info``'s ``comfy_target`` block does; suppressing it for a loopback
    host would trade a false alarm for a silent one on the tunnel case.

    Conservative on all three axes:

    * **Nothing configured** (:func:`_comfy_target` is ``None``) → the payload is
      returned unchanged, the same object, so the local default stays
      byte-identical.
    * **Malformed config** → reported, never raised. A bad ``COMFYUI_URL`` raises
      out of :func:`_comfy_target`, and these two tools never touch the remote,
      so it must not break them — the same "local behavior unchanged" contract
      :func:`_with_target` honors. But swallowing it outright would
      make a typo look like "nothing configured", which is the one reading this
      key exists to rule out, so it becomes an ERROR-SHAPED note (``error`` /
      ``note``) the way ``server_info`` reports the same breakage. Absence of the
      key then means exactly one thing: no remote is configured.
    * **Foreign payload shape** (not a ``dict``) → returned untouched rather than
      reshaped, mirroring :func:`_drop_cloud_jobs`.

    The key cannot realistically collide — ComfyUI's ``/system_stats`` has only
    ``system`` and ``devices`` at the top level, and ``comfy free`` returns
    comfy-cli's own ``{"requested": ..., "note": ...}`` — but if one ever
    appeared, the existing value wins: annotation must never clobber engine data.
    """
    if not isinstance(payload, dict) or _COMFY_TARGET_NOTE_KEY in payload:
        return payload
    try:
        target = _comfy_target()
    except ComfyCliError as exc:
        return {**payload, _COMFY_TARGET_NOTE_KEY: _malformed_target_note(exc)}
    if target is None:
        return payload
    host, port, source = target
    # Shallow copy rather than an in-place insert: reshaping comfy-cli's parsed
    # payload under the caller is not this helper's call (see _drop_cloud_jobs).
    return {
        **payload,
        _COMFY_TARGET_NOTE_KEY: {
            "host": _redact_target_host(host),
            "port": port,
            "source": source,
            "note": (
                "this payload describes whichever ComfyUI comfy-cli itself "
                "targets — `comfy system-stats` / `comfy free` take no "
                "--host/--port — and NOT necessarily the host/port above, which "
                "is where the run/job tools submit. The two are the same machine "
                "when that host resolves to THIS box and different machines "
                "otherwise; this server does not verify which, so compare them "
                "before gating a run on these numbers, and ask the user when the "
                "host is one you cannot place (a loopback host can be a tunnel "
                "to a remote GPU)."
            ),
        },
    }


def _target_provenance_suffix() -> str:
    """The same divergence :func:`_annotate_comfy_target` carries, for a FAILURE.

    The note above only ever reaches a caller on the SUCCESS path, and the
    common case with a remote configured is the failing one: ``comfy
    system-stats`` / ``comfy free`` take no ``--host`` / ``--port``, so with no
    local ComfyUI running they raise comfy-cli's bare ``server_not_running``
    while the configured remote is up and serving the run/job tools perfectly
    well. An agent reading that error alone concludes the remote is down and
    abandons a ``run_workflow`` that would have worked — the provenance is
    exactly as load-bearing on the error as it is on the payload.

    Returns a suffix to append to the raised message, or ``""`` when nothing is
    configured — in which case the caller must re-raise the original objects
    untouched, so the local default stays byte-identical.

    Fail-soft on a malformed config for the same reason
    :func:`_annotate_comfy_target` is: these two tools never touch the remote,
    so a bad ``COMFYUI_URL`` must not make them fail differently. But it does
    not vanish either — a typo must not read as "nothing configured" — so it
    becomes its own short suffix carrying :func:`_comfy_target`'s own
    (already userinfo-masked) message, the same rationale as
    :func:`_malformed_target_note`.

    It does not ADJUDICATE, exactly as the success-path note does not: the two
    ends are the same machine whenever the configured host resolves to this box
    and this server cannot tell whether it does. What it rules out is the one
    wrong inference — that this error is a verdict on the configured target.
    """
    try:
        target = _comfy_target()
    except ComfyCliError as exc:
        # Names all three knobs rather than the two that resolve a host: a lone
        # `COMFYUI_PORT` raises out of `_comfy_target` too, and blaming
        # `COMFYUI_URL`/`COMFYUI_HOST` for it points at variables that are not
        # even set. The embedded message says which one it actually was.
        return (
            " (note: the remote-target config (COMFYUI_URL / COMFYUI_HOST / "
            "COMFYUI_PORT) is set but invalid, so no remote is resolved and this "
            "error is comfy-cli's own, about whichever ComfyUI it itself targets; "
            f"the config error is: {exc})"
        )
    if target is None:
        return ""
    host, port, source = target
    endpoint = _format_target_endpoint(_redact_target_host(host), port)
    # "not AIMED at it", not "did not REACH it": the absence of --host/--port is
    # a fact about what this invocation asked for, whereas whether it reached
    # that endpoint anyway is exactly what this server cannot know (the host may
    # resolve to this box, `COMFY_LOCAL_URL` may point comfy-cli straight at it).
    # Claiming non-reach would misclassify a genuine failure OF the target as
    # unrelated to it — the same over-reach the closing clause avoids by saying
    # this failure is not evidence either way rather than that the remote is up.
    return (
        f" (note: {source} is set to {endpoint}, but this probe was NOT aimed at "
        "it — `comfy system-stats` / `comfy free` take no --host/--port, so this "
        "error is about whichever ComfyUI comfy-cli itself targets. That may or "
        "may not be the same machine, and this server does not verify which, so "
        "do not read this as a verdict on the configured target: that is where "
        "the run/job tools submit, and this failure is no evidence about it.)"
    )


def _comfy_cli_ran(err: ComfyCliError) -> bool:
    """True when *err* is a verdict from a comfy-cli child that actually RAN.

    ``returncode`` is set wherever :func:`_unwrap_envelope` read a child's exit
    status (the error-envelope path and the no-envelope path alike), and
    ``timed_out`` marks the child we killed at our own deadline; neither can be
    set unless comfy-cli was spawned.

    The failures that leave both at their defaults are the ones raised about
    this machine's INSTALL rather than about any ComfyUI —
    :func:`_require_comfy_bin`'s missing binary and its macOS TCC denial,
    :func:`_check_comfy_version`'s version floor, :func:`_unwrap_envelope`'s
    refusal of an incompatible envelope schema. :func:`_target_provenance_suffix`
    must not be appended to those: it says this failure is not a verdict on the
    configured target, implying the submit tools are unaffected, and every one of
    them breaks ``run_workflow`` in exactly the same way — there is no binary for
    it to shell out to either.
    """
    return err.returncode is not None or err.timed_out


def _with_target_provenance(err: ComfyCliError) -> ComfyCliError:
    """*err* with :func:`_target_provenance_suffix` appended, or *err* itself.

    Returning the SAME object when no remote is configured is the contract: the
    unconfigured path must re-raise exactly what it raises today, message and
    identity included. Every attribute is carried across rather than only the
    two :func:`_resource_verb_upgrade_error` needs — a timeout here really can
    set ``timed_out`` (both tools pass a ``timeout=``), and a message rewrite is
    no reason for the structured provenance to decay.

    A failure comfy-cli never got to run is returned the same untouched way, for
    the reason :func:`_comfy_cli_ran` gives: the note is about which ComfyUI a
    dispatched verb spoke to, and a missing or unusable comfy-cli spoke to none.
    """
    if not _comfy_cli_ran(err):
        return err
    suffix = _target_provenance_suffix()
    if not suffix:
        return err
    return ComfyCliError(
        f"{err}{suffix}",
        code=err.code,
        no_envelope=err.no_envelope,
        returncode=err.returncode,
        timed_out=err.timed_out,
        data=err.data,
    )
