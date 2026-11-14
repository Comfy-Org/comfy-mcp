"""The MCP Apps inline output viewer (``ui://comfy-mcp/output-viewer``).

A leaf module — it imports the MCP SDK and the standard library, never
``server`` — that owns the one UI resource this server ships. ``server`` builds
the :class:`~mcp.server.apps.Apps` extension from :func:`build_apps`, passes it
to ``MCPServer(..., extensions=[...])``, and stamps :data:`TOOL_META` onto the
two result tools (``fetch_outputs``, ``preview_media``) so an Apps-capable host
renders their output inline.

Why a self-contained HTML string rather than a bundled asset: this repo has no
frontend toolchain, and a ``ui://`` resource is one document the host drops into
a sandboxed iframe. The document reads the tool result the host delivers over
``postMessage`` and renders it with DOM text and ``data:`` URLs only — it opens
no socket, calls no tool, and runs no ``fetch``. The MCP Apps contract keeps the
server blind to the iframe's ``ui/*`` traffic, so there is nothing here but the
bytes.

Graceful degradation is the whole point of registering the resource and the
``_meta.ui`` pointer UNCONDITIONALLY: a host that never negotiated the extension
ignores the field and shows the tool's existing text/image ``content``, so the
viewer is strictly additive. The same ``content`` block — the metadata payload
and any bounded inline image — is what the model reads either way; the iframe is
for the person.

CSP is deliberately left to the ext-apps DEFAULT policy (``default-src 'none'``;
``img-src``/``media-src`` allow ``'self'`` and ``data:``; ``connect-src`` and
``frame-src`` allow nothing). The viewer needs no connect or resource domains —
it renders ``data:`` URLs and text — so declaring any would only widen the
sandbox past what it uses. See :func:`build_apps`.
"""

from __future__ import annotations

from mcp.server.apps import APP_MIME_TYPE, Apps

# The resource URI the two result tools point at via ``_meta.ui.resourceUri``.
# The ``ui://`` scheme is what the SDK (and hosts) key the extension off; the
# path is this server's namespace plus the widget name.
OUTPUT_VIEWER_URI = "ui://comfy-mcp/output-viewer"

# Human label the host shows for the resource / the rendered panel.
OUTPUT_VIEWER_TITLE = "Output preview"

# The MIME a ``ui://`` HTML resource is served as. Re-exported from the SDK so
# this module's constant and the bytes ``add_html_resource`` actually writes
# cannot drift; the viewer test pins the served resource to this value.
OUTPUT_VIEWER_MIME = APP_MIME_TYPE

# The ``_meta`` block to pass to ``@mcp.tool(meta=...)`` on the result tools.
# The SDK serializes ``Tool.meta`` as ``_meta`` on the wire, so this lands as
# ``_meta.ui.resourceUri`` in ``tools/list`` — the Apps pointer — without moving
# the tool onto a second decorator. Stamped on ``fetch_outputs`` and
# ``preview_media`` ONLY; every other tool leaves ``_meta`` unset.
TOOL_META = {"ui": {"resourceUri": OUTPUT_VIEWER_URI}}

# The ext-apps iframe<->host bridge protocol revision. Apps does not require the
# 2026-07-28 core MRTR work (hosts advertise the extension inside `initialize`),
# so the viewer is independent of that; this is the UI bridge's own version.
_UI_PROTOCOL_VERSION = "2026-01-26"

# The self-contained document. It speaks the ext-apps postMessage bridge:
# announces `ui/initialize` at the protocol version above, then
# `ui/notifications/initialized`; handles the `tool-result` delivery, the
# `host-context-changed` theme signal, and `ui/resource-teardown`; and reports
# its height with `ui/notifications/size-changed`. It never calls a tool, never
# fetches, and never touches `innerHTML` — DOM text goes through `textContent`
# and media through `data:` URLs, so the default CSP is all it needs.
#
# Rendering model: a `content[]` block of `type: "image"` carries base64 the
# host already delivered (what `fetch_outputs(inline_images=True)` and
# `preview_media` produce today), rendered as a `data:` URL. Once structured
# file records land, a `structuredContent.files[]` entry
# `{name, path, media_type, bytes, data_url?}` renders its `data_url` only when
# it is a `data:` URL, with the path and size shown as text; a record with no
# inlinable bytes shows the on-disk path and says the file was not inlined.
OUTPUT_VIEWER_HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{OUTPUT_VIEWER_TITLE}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{
    margin: 0;
    padding: 12px;
    font: 13px/1.5 system-ui, -apple-system, sans-serif;
  }}
  .item {{ margin: 0 0 16px; }}
  .item:last-child {{ margin-bottom: 0; }}
  .media {{
    display: block;
    max-width: 100%;
    height: auto;
    border-radius: 6px;
  }}
  .meta {{ margin-top: 4px; opacity: 0.7; word-break: break-all; }}
  .empty {{ opacity: 0.7; }}
</style>
</head>
<body>
<main id="root"></main>
<script>
(function () {{
  "use strict";
  var PROTOCOL_VERSION = {_UI_PROTOCOL_VERSION!r};
  var root = document.getElementById("root");
  var reqId = 0;

  function post(message) {{
    // The iframe origin is opaque under the sandbox, so target "*"; the host
    // validates the source frame on its side.
    if (window.parent && window.parent !== window) {{
      window.parent.postMessage(message, "*");
    }}
  }}

  function notify(method, params) {{
    post({{ jsonrpc: "2.0", method: method, params: params || {{}} }});
  }}

  function reportSize() {{
    notify("ui/notifications/size-changed", {{
      height: document.documentElement.scrollHeight
    }});
  }}

  function clear() {{
    while (root.firstChild) {{ root.removeChild(root.firstChild); }}
  }}

  function addText(parent, className, text) {{
    var el = document.createElement("div");
    el.className = className;
    el.textContent = text;
    parent.appendChild(el);
    return el;
  }}

  function isDataUrl(value) {{
    return typeof value === "string" && value.indexOf("data:") === 0;
  }}

  // Build a `data:` URL from an image `content[]` block. The host delivers the
  // base64 the tool already produced; mirror it back into an <img>.
  function imageDataUrl(block) {{
    if (!block || isDataUrl(block.data)) {{ return block ? block.data : null; }}
    if (typeof block.data !== "string") {{ return null; }}
    var mime = block.mimeType || "image/png";
    return "data:" + mime + ";base64," + block.data;
  }}

  function renderMedia(parent, mediaType, dataUrl) {{
    // Only ever a `data:` URL reaches here — never a path, a file: URL, or a
    // loopback http URL — so the default CSP (img/media `data:`) covers it.
    if (!isDataUrl(dataUrl)) {{ return false; }}
    var kind = (mediaType || "").split("/")[0];
    var tag = kind === "video" ? "video" : kind === "audio" ? "audio" : "img";
    var el = document.createElement(tag);
    el.className = "media";
    if (tag !== "img") {{ el.controls = true; }}
    el.src = dataUrl;
    parent.appendChild(el);
    return true;
  }}

  function renderImageBlock(block) {{
    var item = document.createElement("div");
    item.className = "item";
    renderMedia(item, block.mimeType || "image/png", imageDataUrl(block));
    root.appendChild(item);
  }}

  function renderFileRecord(file) {{
    var item = document.createElement("div");
    item.className = "item";
    var shown = file && isDataUrl(file.data_url)
      ? renderMedia(item, file.media_type, file.data_url)
      : false;
    if (!shown) {{
      addText(item, "empty", "File on disk — not inlined.");
    }}
    var meta = [];
    if (file && file.name) {{ meta.push(String(file.name)); }}
    if (file && file.path) {{ meta.push(String(file.path)); }}
    if (file && typeof file.bytes === "number") {{
      meta.push(file.bytes + " bytes");
    }}
    if (meta.length) {{ addText(item, "meta", meta.join(" \\u00b7 ")); }}
    root.appendChild(item);
  }}

  function render(result) {{
    clear();
    var rendered = 0;
    var content = (result && result.content) || [];
    var i;
    for (i = 0; i < content.length; i++) {{
      if (content[i] && content[i].type === "image") {{
        renderImageBlock(content[i]);
        rendered++;
      }}
    }}
    var structured = result && result.structuredContent;
    var files = structured && structured.files;
    if (Array.isArray(files)) {{
      for (i = 0; i < files.length; i++) {{
        renderFileRecord(files[i]);
        rendered++;
      }}
    }}
    if (!rendered) {{
      addText(root, "empty", "No inline preview available.");
    }}
    reportSize();
  }}

  function applyTheme(context) {{
    var theme = context && context.theme;
    // Explicit light/dark wins; anything else falls back to the CSS default
    // (`color-scheme: light dark`), which already follows the host.
    if (theme === "light" || theme === "dark") {{
      document.documentElement.style.colorScheme = theme;
    }} else {{
      document.documentElement.style.removeProperty("color-scheme");
    }}
  }}

  // The host can hand the first tool result either as the `ui/initialize`
  // response or as a follow-up `tool-result` notification; accept both shapes.
  function extractToolResult(payload) {{
    if (!payload) {{ return null; }}
    if (payload.toolResult) {{ return payload.toolResult; }}
    if (payload.content || payload.structuredContent) {{ return payload; }}
    return null;
  }}

  window.addEventListener("message", function (event) {{
    var msg = event.data;
    if (!msg || typeof msg !== "object") {{ return; }}
    var method = msg.method;
    if (method === "ui/notifications/tool-result") {{
      var tr = extractToolResult(msg.params);
      if (tr) {{ render(tr); }}
      return;
    }}
    if (method === "host-context-changed" ||
        method === "ui/notifications/host-context-changed") {{
      applyTheme(msg.params && (msg.params.hostContext || msg.params));
      return;
    }}
    if (method === "ui/resource-teardown") {{
      clear();
      return;
    }}
    // The `ui/initialize` response: id-matched, may carry initial render data
    // and host context.
    if (msg.id !== undefined && msg.result) {{
      applyTheme(msg.result.hostContext);
      var initial = extractToolResult(msg.result);
      if (initial) {{ render(initial); }}
    }}
  }});

  // Kick off the handshake.
  reqId++;
  post({{
    jsonrpc: "2.0",
    id: reqId,
    method: "ui/initialize",
    params: {{ protocolVersion: PROTOCOL_VERSION }}
  }});
  notify("ui/notifications/initialized", {{}});
  reportSize();
}})();
</script>
</body>
</html>
"""


def build_apps() -> Apps:
    """The configured :class:`~mcp.server.apps.Apps` extension to hand ``MCPServer``.

    Registers the single output-viewer resource and nothing else. No ``csp=`` is
    passed, so the SDK writes no ``_meta.ui.csp`` and the host applies the
    ext-apps DEFAULT policy — which already allows the ``data:`` URLs and text
    this viewer uses while forbidding every connect/frame origin it does not. The
    ``_meta.ui.resourceUri`` pointer is stamped on the tools in ``server`` via
    :data:`TOOL_META` rather than through ``apps.tool`` so the existing
    ``@mcp.tool()`` result tools keep their one decorator.
    """
    apps = Apps()
    apps.add_html_resource(
        OUTPUT_VIEWER_URI,
        OUTPUT_VIEWER_HTML,
        name=OUTPUT_VIEWER_URI,
        title=OUTPUT_VIEWER_TITLE,
    )
    return apps
