"""Tests for transport selection and the HTTP bind's loopback default.

``comfy-mcp`` served stdio and only stdio until comfy-mcp#273 asked for a
Streamable HTTP listener, so these cover two different kinds of claim:

1. **Nothing changed for anyone who did not ask.** An empty environment, and an
   environment that sets the HTTP knobs but not the transport, both still
   resolve to plain stdio with no keyword arguments — the exact call the entry
   point made before. This is the assertion that keeps the feature free.
2. **The bind is loopback unless the USER opted out**, which is the decision
   #273 settled. The opt-in is an environment variable, so it can only be set by
   whoever launches the process; a prompt-injected caller has no route to it,
   which is why there is no elicitation here to mirror the ``--listen`` gate's.

The bad-value cases assert a **named refusal rather than a fallback**. A typo'd
transport that quietly served stdio, or a port the SDK rejected deep inside
uvicorn, would both present as "it starts and nothing connects".
"""

from __future__ import annotations

import pytest

from comfy_mcp import errors, server

# Every spelling that must reach the Streamable HTTP transport. `http` is the
# alias people type; the SDK's own name is the long one.
_HTTP_SPELLINGS = ["http", "HTTP", "streamable-http", "Streamable-HTTP"]

# Addresses that name only this machine, in the spellings a user actually
# writes. `[::1]` is included because that is how a v6 literal is copied out of
# a URL, and `_address_is_loopback` unwraps the brackets.
_LOOPBACK_HOSTS = ["127.0.0.1", "127.0.0.53", "localhost", "::1", "[::1]"]

# Addresses that publish the listener. `0.0.0.0` and `::` are the two every
# "make it reachable" tutorial reaches for; the rest are the fail-closed cases —
# a routable literal, and a DNS name `_address_is_loopback` will not resolve.
_REMOTE_HOSTS = ["0.0.0.0", "::", "10.0.0.5", "192.168.1.20", "comfy.internal"]


def test_empty_environment_serves_stdio_with_no_kwargs():
    """The pre-#273 call, unchanged: `mcp.run(transport="stdio")` and nothing else."""
    assert server._resolve_transport({}) == ("stdio", {})


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_transport_is_treated_as_unset(value):
    """An `env` block that sets the variable to nothing is not a request for HTTP."""
    assert server._resolve_transport({server._TRANSPORT_ENV: value}) == ("stdio", {})


def test_stdio_ignores_the_http_knobs_rather_than_refusing():
    """Host and port set without the transport bind nothing, so they are ignored.

    Asserted explicitly because the alternative — refusing to start on a
    leftover variable — would turn a harmless misconfiguration into an outage,
    and because the combination is reachable by exactly one mistake (setting the
    host and forgetting the transport) whose outcome is already the safe one.
    """
    env = {
        server._TRANSPORT_ENV: "stdio",
        server._HTTP_HOST_ENV: "0.0.0.0",
        server._HTTP_PORT_ENV: "9999",
    }
    assert server._resolve_transport(env) == ("stdio", {})


@pytest.mark.parametrize("spelling", _HTTP_SPELLINGS)
def test_http_spellings_resolve_to_the_sdk_transport_name(spelling):
    transport, kwargs = server._resolve_transport({server._TRANSPORT_ENV: spelling})
    assert transport == "streamable-http"
    assert kwargs == {"host": "127.0.0.1", "port": 8000}


def test_sse_is_not_offered():
    """The SDK still has an `sse` transport; this server does not expose it.

    Deprecated in the MCP spec, and nothing asked for it — so it stays out
    rather than becoming a surface that has to keep working.
    """
    with pytest.raises(errors.ComfyCliError) as caught:
        server._resolve_transport({server._TRANSPORT_ENV: "sse"})
    assert "sse" not in str(caught.value).split("accepted:")[1]


@pytest.mark.parametrize("host", _LOOPBACK_HOSTS)
def test_loopback_hosts_need_no_opt_in(host):
    _, kwargs = server._resolve_transport(
        {server._TRANSPORT_ENV: "http", server._HTTP_HOST_ENV: host}
    )
    assert kwargs["host"] == host


@pytest.mark.parametrize("host", _REMOTE_HOSTS)
def test_non_loopback_bind_is_refused_without_the_opt_in(host):
    """The #273 decision: loopback by default, explicit opt-in for wider.

    The message is asserted on three things rather than wording: it names the
    variable that caused it, it states the no-authentication stake the decision
    rests on, and it names the variable that permits it. A refusal that fails
    any of those leaves the user guessing.
    """
    with pytest.raises(errors.ComfyCliError) as caught:
        server._resolve_transport(
            {server._TRANSPORT_ENV: "http", server._HTTP_HOST_ENV: host}
        )
    message = str(caught.value)
    assert server._HTTP_HOST_ENV in message
    assert "NO authentication" in message
    assert server._ALLOW_REMOTE_BIND_ENV in message


@pytest.mark.parametrize("host", _REMOTE_HOSTS)
@pytest.mark.parametrize("opt_in", ["1", "true", "TRUE", "yes", "on"])
def test_non_loopback_bind_is_allowed_with_the_opt_in(host, opt_in):
    _, kwargs = server._resolve_transport(
        {
            server._TRANSPORT_ENV: "http",
            server._HTTP_HOST_ENV: host,
            server._ALLOW_REMOTE_BIND_ENV: opt_in,
        }
    )
    assert kwargs["host"] == host


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "  ", "maybe"])
def test_a_negative_or_unrecognized_opt_in_is_not_an_opt_in(value):
    """A set-to-something check is the wrong test for a bind decision.

    `COMFY_MCP_ALLOW_REMOTE_BIND=0` is how a user writes "no", and a presence
    check would read it as "yes" — the one direction of this mistake that
    publishes an unauthenticated listener.
    """
    with pytest.raises(errors.ComfyCliError):
        server._resolve_transport(
            {
                server._TRANSPORT_ENV: "http",
                server._HTTP_HOST_ENV: "0.0.0.0",
                server._ALLOW_REMOTE_BIND_ENV: value,
            }
        )


def test_default_host_is_loopback_without_consulting_the_sdk():
    """The default is stated here, not inherited.

    `run_streamable_http_async` happens to default to `127.0.0.1` today. This
    asserts our own constant, so an SDK upgrade that changed its default could
    not widen this server's bind silently.
    """
    assert server._address_is_loopback(server._DEFAULT_HTTP_HOST)


@pytest.mark.parametrize("bad", ["htp", "streamablehttp", "tcp", "websocket", "-"])
def test_an_unknown_transport_is_named_not_silently_stdio(bad):
    with pytest.raises(errors.ComfyCliError) as caught:
        server._resolve_transport({server._TRANSPORT_ENV: bad})
    message = str(caught.value)
    assert server._TRANSPORT_ENV in message
    # Every accepted spelling is listed, so the fix is in the error.
    for spelling in ("stdio", "streamable-http", "http"):
        assert spelling in message


@pytest.mark.parametrize("port,expected", [("8000", 8000), ("1", 1), ("65535", 65535)])
def test_port_is_read_from_the_environment(port, expected):
    _, kwargs = server._resolve_transport(
        {server._TRANSPORT_ENV: "http", server._HTTP_PORT_ENV: port}
    )
    assert kwargs["port"] == expected


@pytest.mark.parametrize("bad", ["nope", "80.5", "0x50", "", " "])
def test_a_non_integer_port_is_named_or_falls_back_to_the_default(bad):
    """A blank port is "unset"; anything non-numeric is a named refusal.

    Grouped in one test because the boundary between them is the thing worth
    pinning: `""` must not raise (an empty `env` entry is not a value), and
    `"80.5"` must, because it is a value the user believed they had set.
    """
    env = {server._TRANSPORT_ENV: "http", server._HTTP_PORT_ENV: bad}
    if bad.strip():
        with pytest.raises(errors.ComfyCliError) as caught:
            server._resolve_transport(env)
        assert server._HTTP_PORT_ENV in str(caught.value)
    else:
        _, kwargs = server._resolve_transport(env)
        assert kwargs["port"] == server._DEFAULT_HTTP_PORT


@pytest.mark.parametrize("bad", ["0", "-1", "65536", "99999999"])
def test_an_out_of_range_port_is_refused_before_it_reaches_the_sdk(bad):
    """Named here rather than surfacing as a socket error with no variable in it."""
    with pytest.raises(errors.ComfyCliError):
        server._resolve_transport(
            {server._TRANSPORT_ENV: "http", server._HTTP_PORT_ENV: bad}
        )
