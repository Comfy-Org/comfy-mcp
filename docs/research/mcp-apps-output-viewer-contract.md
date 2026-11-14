# Shared MCP Apps output viewer contract

**Status:** contract only. Neither server emits this payload yet.
**Date:** 2026-11-14.

`comfy-mcp` (this repo: local ComfyUI, Python, stdio) and `Comfy-Org/comfy-cloud-mcp-server` (hosted, TypeScript) each serve their own HTML. They share the tool metadata, the result object, and the rules below, so an Apps host can show a finished image, video, audio file, or 3D asset from either server.

This is groundwork for one output surface across the two MCP servers. A milestone architecture TDD for that convergence was not among the sources that could be read, so this file does not restate it.

The local spike on `cursor/mcp-spec-audit-apps-spike-b235` (`docs/research/mcp-spec-audit-and-apps-spike.md`) proposed `ui://comfy-mcp/output-viewer` and a file record `{name, path, media_type, bytes, data_url?}`. The item in this contract replaces that record. The spike's security choices stand: local bytes travel as `data:` URLs, and the view does not fetch or point at loopback.

## What was checked

Cloud PRs [1894](https://github.com/Comfy-Org/comfy-cloud-mcp-server/pull/1894) (merged 2026-11-11) and [2005](https://github.com/Comfy-Org/comfy-cloud-mcp-server/pull/2005) (merged 2026-11-13) were read via the GitHub API, as were `src/output-viewer-resource.ts`, `src/mcp-server.ts`, `src/tools-registration.test.ts`, and `src/job-output-shared.ts` on cloud `main`. PR 1894 shipped the viewer. PR 2005 extracted the shared `postMessage` bridge into `src/mcp-app-bridge.ts` and did not change rendering. Cloud `main` has since attached `wait_for_job` to the same view and taught the view to call `get_output`.

Local inline limits are `_INLINE_IMAGE_MAX_COUNT = 8` and `_INLINE_IMAGE_MAX_BYTES = 16 * 1024 * 1024` in `src/comfy_mcp/server.py`, shared by `fetch_outputs` and `preview_media`. The Apps wire rules are the 2026-01-26 apps specification (`ui://`, `_meta.ui.resourceUri`, MIME `text/html;profile=mcp-app`). Python registration is `mcp.server.apps` (`Apps.add_html_resource`, `Apps.tool`, `APP_MIME_TYPE`, `client_supports_apps`).

## Resource URI

The servers do not share one URI. The host loads the HTML from the server that advertised it and builds the iframe CSP from that resource. Local media is a `data:` URL and cloud media is `https`, so the document and the origin list differ.

The path is the shared name. The authority names the server.

| Server | URI | MIME |
| --- | --- | --- |
| Local | `ui://comfy-mcp/output-viewer` | `text/html;profile=mcp-app` |
| Cloud | `ui://comfy-cloud/output-viewer` | `text/html;profile=mcp-app` |

Cloud already registers the second URI. Resource name on both is `output-viewer`. A further view uses `ui://<server>/output-viewer`, with no version segment.

Tool metadata uses the nested field (the flat `_meta["ui/resourceUri"]` form is the deprecated one in the apps spec):

```json
{ "_meta": { "ui": { "resourceUri": "ui://comfy-mcp/output-viewer" } } }
```

Cloud substitutes its own URI. Local registers with `mcp.server.apps.Apps`: `add_html_resource` for the document, and the URI on the tool via `@apps.tool(resource_uri=...)` or via `meta` on the existing `@mcp.tool`, not both. The SDK rejects a tool whose URI has no resource, and a second decorator would register the tool twice. `client_supports_apps` may gate the optional note in the text fallback. It does not gate registration.

## Tools that carry `_meta.ui`

A tool carries the URI when a successful result's `structuredContent` is the object below. Every other tool leaves `_meta.ui` unset.

| Server | Tools |
| --- | --- |
| Local | `fetch_outputs`, `preview_media` |
| Cloud | `get_output`, `get_batch_output` |

`run_workflow`, `run_template`, and `partner_generate` return a job. The model then calls `fetch_outputs`. Cloud `wait_for_job` returns status, not media items, so it does not carry the viewer. `visibility` stays omitted, so these tools stay model-visible. This contract adds no app-only tool.

Register the resource for every profile that registers those tools, except cloud's `chatgpt` profile, which stays resource-free. Do not hide the resource from other hosts based on the Apps extension. `resources/list` stays stable for a process.

## Result payload

The host delivers the tool result to the iframe. The viewer reads `structuredContent.items` and nothing else. `url`, when present, is a `data:` URL (local bytes) or an `https` URL (cloud). Each server emits only the form it can host. Keys are snake_case, like the rest of both tool payloads.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "mcp-output-viewer/v1.json",
  "title": "OutputViewerResult",
  "type": "object",
  "additionalProperties": true,
  "required": ["contract_version", "items"],
  "properties": {
    "contract_version": { "const": 1 },
    "items": {
      "type": "array",
      "items": { "$ref": "#/$defs/mediaItem" }
    }
  },
  "$defs": {
    "mediaItem": {
      "type": "object",
      "additionalProperties": true,
      "required": ["mime_type", "kind", "filename", "size_bytes", "prompt_id"],
      "properties": {
        "url": {
          "anyOf": [
            {
              "type": "string",
              "pattern": "^data:[a-zA-Z0-9.+-]+/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/]+=*$"
            },
            { "type": "string", "pattern": "^https://[^\\s]+$" }
          ]
        },
        "mime_type": { "type": "string", "minLength": 1 },
        "kind": { "enum": ["image", "video", "audio", "3d"] },
        "filename": { "type": "string", "minLength": 1 },
        "size_bytes": { "type": "integer", "minimum": 0 },
        "prompt_id": { "type": "string", "minLength": 1 }
      }
    }
  }
}
```

Extra properties are allowed and ignored, so a server can keep a warning or a batch count beside `items` during a transition. `url` is optional: absent means the bytes are not in the result. `size_bytes` is the media length, never the base64 length. For a `data:` URL it is the decoded length and it counts toward the cap below. `prompt_id` is the job id, repeated on every item so a batch card can name its own job. `kind` is the server's classification; the viewer does not sniff the filename. `3d` is a filename row in version 1: it is not assigned to `img`, `video`, or `audio`.

`image`, `video`, and `audio` with a valid `url` become that element. `src` is set only after the string matches the schema, the `data:` media type matches `mime_type`, and an `https` origin is in that server's `resourceDomains`. Local declares none, so local HTML does not assign an `https` `src`.

Local, one inlined PNG from `fetch_outputs`. `preview_media` uses the same item for the contact sheet or waveform PNG it writes:

```json
{
  "contract_version": 1,
  "items": [
    {
      "url": "data:image/png;base64,iVBORw0KGgo=",
      "mime_type": "image/png",
      "kind": "image",
      "filename": "ComfyUI_00001_.png",
      "size_bytes": 2048,
      "prompt_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    }
  ]
}
```

Cloud, one image from `get_output`. A `get_batch_output` item is the same object with that item's `prompt_id`:

```json
{
  "contract_version": 1,
  "items": [
    {
      "url": "https://cloud.comfy.org/api/s/example",
      "mime_type": "image/png",
      "kind": "image",
      "filename": "ComfyUI.png",
      "size_bytes": 842311,
      "prompt_id": "7c1e0000-0000-4000-8000-000000000001"
    }
  ]
}
```

A local video past the cap has no `url`. The text block carries the absolute path and the size. The card shows filename, kind, and size:

```json
{
  "contract_version": 1,
  "items": [
    {
      "mime_type": "video/mp4",
      "kind": "video",
      "filename": "ComfyUI_00001_.mp4",
      "size_bytes": 48000000,
      "prompt_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    }
  ]
}
```

## Text fallback

`content[]` stays the complete result for hosts that ignore `_meta.ui`, and for the model.

- Local text keeps today's metadata, including on-disk paths. `inline_images=True` and `preview_media` may still append `ImageContent` under the cap. The Apps object does not replace those blocks. Paths stay, because the next tool call needs them.
- Cloud text keeps today's link and download command, including an `/api/s/<id>` shortlink when one was minted.

When the client advertised `io.modelcontextprotocol/ui` with MIME `text/html;profile=mcp-app`, cloud may append one line saying the viewer is showing the media and not to paste signed storage URLs. The line is extra. It does not remove the shortlink or the command. Other hosts get today's text unchanged.

A `contract_version` the HTML does not understand still leaves this text in place. The iframe then lists filename, kind, and size, and does not guess a media element.

## Security

The document is one self-contained HTML string.

- DOM nodes come from `createElement`. Human-readable strings go through `textContent`.
- `img`, `video`, and `audio` `src` is set only from a `url` that passed the checks above.
- The bridge sends `ui/initialize` at protocol `2026-01-26`, then `ui/notifications/initialized`, then `ui/notifications/size-changed`. It handles `tool-result`, `host-context-changed` (`color-scheme`), and `ui/resource-teardown`.
- Messages go to `window.parent` and are accepted only when `event.source` is `window.parent`.
- Download may call `ui/open-link` only with an `https` URL the server already treats as user-facing. Local has none. Cloud's is an `/api/s/<id>` shortlink with an empty query or exactly `?raw=1`. A signed storage URL may be the media `src` and is not passed to `ui/open-link`.

The document contains no `fetch`, `XMLHttpRequest`, `WebSocket`, `innerHTML`, `insertAdjacentHTML`, `document.write`, or `eval`. It makes no `tools/call`. It requests no camera, microphone, geolocation, or clipboard permission, and it does not load `file://` or loopback.

CSP on the resource `_meta.ui` and on `resources/read`:

| Server | `connectDomains` | `resourceDomains` |
| --- | --- | --- |
| Local | omitted | omitted |
| Cloud | omitted | `https://storage.googleapis.com`, the session's public API origin, `https://cloud.comfy.org`, `https://stagingcloud.comfy.org` |

Omitted local CSP is the apps default: `img-src` and `media-src` allow `'self'` and `data:`, and `connect-src` is `'none'`. The host template also allows `data:` when `resourceDomains` is set, so cloud can ignore a `data:` URL without a CSP change. No `frameDomains`, no `permissions`, no dedicated sandbox `domain`.

The host renders the HTML in a sandboxed iframe. Web hosts use the spec's outer sandbox proxy plus an inner frame. Inline script depends on the default `script-src 'self' 'unsafe-inline'`. The view assumes no other permission.

## Size limits

`_INLINE_IMAGE_MAX_BYTES` is 16 MiB of raw file bytes and `_INLINE_IMAGE_MAX_COUNT` is 8. Both already bound `fetch_outputs(inline_images=True)` and `preview_media`. Video and audio that join the inline result use this same budget.

For `data:` URLs in one tool result:

| Bound | Value | What counts |
| --- | --- | --- |
| One item | 16 MiB | that item's `size_bytes` |
| The result | 16 MiB | sum of `size_bytes` over items that have a `data:` URL |
| The result | 8 items | items that have a `data:` URL |

`https` items do not count. A `data:` item over 16 MiB, one that would push the sum over 16 MiB, or a ninth `data:` item is emitted without `url`, in order, stopping at the first item that would break a bound. The on-disk file is unchanged. The text names the path and the size.

Base64 expands the wire size by about 4/3, so a full 16 MiB inline is on the order of 22 MiB of JSON. A host that truncates the tool result can drop the URL. The text fallback remains, and a missing or truncated `url` is shown as not inlined.

Cloud keeps its character ceiling for `https` results: `MAX_GET_OUTPUT_RESULT_CHARS` is 40,000 (text plus compact `structuredContent`), and one `get_output` page lists at most 25 files (`GET_OUTPUT_PAGE_MAX_FILES`). A URL dropped by that ceiling leaves the item in `items` without `url`, and the text says how to fetch the next page. Cloud does not embed `data:` URLs under this contract, so the 16 MiB cap applies there only if a later change inlines bytes.

## Versioning

`contract_version` is an integer on the result. This document is version 1. The URI does not carry it. One `output-viewer` resource per server; the HTML reads the integer.

An additive change stays on version 1: a new optional field, or a new `kind` the current HTML shows as a filename row. Viewers ignore unknown fields. A breaking change (renaming `url`, changing what a `kind` means, adding a required field) bumps the integer. The server keeps emitting 1 until the HTML it serves understands the new number. A higher number the HTML does not understand becomes the filename list plus the text fallback.

A new URI (`ui://comfy-mcp/output-viewer-v2` and the cloud twin) is for the case where the HTML and the payload must move together and a host could cache the old document against the new result. The HTML `appInfo.version` is not `contract_version`.

## Changes the cloud viewer needs to conform

Against PR 1894, PR 2005, and cloud `main` on 2026-11-14. PR 2005 only shares the bridge string. Keep `mcpAppBridgeJs` when the HTML changes.

1. **Keep** `ui://comfy-cloud/output-viewer`. The authority stays `comfy-cloud`.
2. **Read `structuredContent.items`** and require `contract_version === 1`. Stop selecting `results[]` for `get_output` and `outputs[]` for `get_batch_output`. Batch `summary`, `failed`, and `pending` may stay as extra fields; the media loop does not read them.
3. **Emit** `url`, `mime_type`, `kind`, `filename`, `size_bytes`, and `prompt_id`. Stop classifying with the extension maps (`png`, `jpg`, `mp4`, `mp3`, …). `kind` is `image`, `video`, `audio`, or `3d`. `mime_type` is an IANA type. The outputs-map key (`images`, `gifs`, …) and `file.type` (`output` / `temp`) are not MIME types; copy neither into `mime_type`.
4. **One `url`, `https` only.** Stop evaluating `inline_url ?? url`, and stop accepting `http:`. The media `src` is `url`. Download still uses `ui/open-link` only for an `/api/s/<id>` shortlink with an empty query or exactly `?raw=1`. A signed `https://storage.googleapis.com` URL may be `url` when that is what should play.
5. **`prompt_id` on every item.** `get_output` copies the job id onto each item. `get_batch_output` stores each entry's job id in `prompt_id` (today's `job_id`).
6. **`size_bytes` on every item**, from a length the server already trusts. Today's payload has `saved_bytes` only after a disk save, plus `width` and `height`. Pixel size is not a byte length.
7. **Drop `url` instead of setting `url_omitted: true`.** When the 40,000-character ceiling or the 25-file page removes a download URL, the item stays, without `url`, and still has `mime_type`, `kind`, `filename`, `size_bytes`, and `prompt_id`. The text note and the offset redirect stay.
8. **Attach the URI only on `get_output` and `get_batch_output`.** `main` also sets it on `wait_for_job`, and the view calls `tools/call` `get_output` when that status says completed, and again from Refresh. PR 1894 left `wait_for_job` alone and made no tool call. The conforming view renders the pushed result and does not call a tool. A finished wait stays a status line; the model calls `get_output`.
9. **Reject a `url` that fails the schema** before setting `src` (`http:`, `javascript:`, `file:`). Cloud does not emit `data:`. A media element that fails to load still becomes "Preview unavailable" through `textContent`.
10. **Keep the security pins:** no `fetch`, no `innerHTML`, no `connectDomains`, text via `textContent`, `chatgpt` resource-free. The upload widget, catalog picker, run-settings panel, and the iterate buttons are other apps. They stay outside this contract and do not change `items`.

## Non-goals

This file does not choose the local HTML, its tests, or the `mcp` package floor. It does not vendor the cloud viewer into this repo. A local implementation stays a thin MCP surface: no HTTP client, and no code copied from the cloud server.
