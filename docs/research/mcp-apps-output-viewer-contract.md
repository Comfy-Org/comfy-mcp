# Shared MCP Apps output viewer contract

**Status:** contract only. `items` is the opt-in viewer shape. Cloud already publishes `results` and `outputs`, and this contract keeps those arrays.
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

A tool carries the URI when its success payload uses the object below. Every other tool leaves `_meta.ui` unset. The metadata is part of tool registration, so a failed call still opens the viewer.

| Server | Tools |
| --- | --- |
| Local | `fetch_outputs`, `preview_media` |
| Cloud | `get_output`, `get_batch_output`, and `wait_for_job` |

`run_workflow`, `run_template`, and `partner_generate` return a job. The model then calls `fetch_outputs`. `wait_for_job` is the cloud exception in the section below: its payload is status, and the iframe may then call `get_output`. `visibility` stays omitted, so these tools stay model-visible. This contract adds no app-only tool.

Register the resource for every profile that registers those tools. Cloud's `chatgpt` profile is the exception: it registers no resource, and `get_output`, `get_batch_output`, and `wait_for_job` omit `_meta.ui` there. Do not hide the resource from other hosts based on the Apps extension. `resources/list` stays stable for a process.

## Result payload

The host delivers the tool result to the iframe. `url` is the play source: a `data:` URL (local bytes) or an `https` URL (cloud). `download_url`, when present, is a separate `https` URL for Download. Each server emits only the form it can host. Keys are snake_case, like the rest of both tool payloads.

The viewer picks one array:

- `contract_version` is 1 and `items` is an array: render `items`, and do not also render `results` or `outputs`.
- `contract_version` is absent, or it is 1 and `items` is not an array: render `results` when that array is present, or `outputs` when that array is present. This is cloud's published agent API.
- `contract_version` is any other number: show the text from `content` and render no media. This wins over `results` and `outputs`, so a future version is not drawn with the version-1 arrays.
- `isError` is true, or none of those arrays is present: show the text from `content` and render no media. An empty chosen array is a finished result with nothing to show.

`results` and `outputs` stay the agent-facing records. `items` is a versioned viewer shape a server may add later. It does not replace those arrays. Putting the same URL strings in both arrays counts them twice toward cloud's 40,000-character ceiling, so a server opts in only by sending `items` without copying those URL strings into `results` or `outputs` in the same payload. If that copy would not fit, the server omits `items` and the viewer stays on the legacy array.

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
    "offset": { "type": "integer", "minimum": 0 },
    "items": {
      "type": "array",
      "items": { "$ref": "#/$defs/mediaItem" }
    }
  },
  "$defs": {
    "mediaItem": {
      "type": "object",
      "additionalProperties": true,
      "required": ["mime_type", "kind", "filename"],
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
        "kind": { "type": "string", "minLength": 1 },
        "filename": { "type": "string", "minLength": 1 },
        "download_url": { "type": "string", "pattern": "^https://[^\\s]+$" },
        "size_bytes": { "type": "integer", "minimum": 0 },
        "prompt_id": { "type": "string", "minLength": 1 },
        "job_id": { "type": "string", "minLength": 1 },
        "url_omitted": { "type": "boolean" },
        "image_block": { "const": true }
      }
    }
  }
}
```

Extra properties are allowed. On a legacy entry the viewer also reads `inline_url`. `url` is optional. Absent means this object does not carry a play URL. `image_block: true` means this item's bytes travel in a paired `ImageContent` block, described below; it is set only on an item with no `url`. With neither a play URL nor `image_block`, the bytes are not in the result. `offset` on the result is the page offset the call was made with; a paged cloud `get_output` result sets it, and absent means 0. `url_omitted: true` means the server dropped the URL to fit the character ceiling. The viewer then ignores `url`, `download_url`, and `inline_url` on that entry and shows the text that says how to ask again. The flag stays; a server does not delete it in favor of a missing `url` alone.

`download_url` is optional and is never the media `src`. `size_bytes` is optional. It is the media length, never the base64 length, and never `0` standing in for unknown. A `data:` URL includes it, because the server measured the file. Omit it when the length was not measured (`fetch_outputs(url_only=True)`, a cloud URL with no known object size, a `getsize` failure). `prompt_id` is the job id when the file belongs to a job. On a batch entry, `job_id` is an alias: if `prompt_id` is absent and `job_id` is a non-empty string, the viewer uses `job_id`. `preview_media` of a path that is not a job output omits both. `kind` is the server's classification on an `items` entry; the viewer does not sniff the filename there. A legacy entry with no `kind` is classified from its filename extension, which is what cloud does today. Version 1 renders `image`, `video`, and `audio` as those elements, and `3d` as a filename row. Any other non-empty `kind` is a filename row too.

Legacy play and download, used when the viewer is on `results` or `outputs`: the play URL is `inline_url` when that is `https`, otherwise `url` when that is `https` or `data:`. Download uses `download_url` when it is present. If `download_url` is absent and `url` is a shortlink, Download uses `url`. A signed storage URL is not a download link.

`image`, `video`, and `audio` with a valid `url` become that element. `src` is set only from that `url`, or from the `data:` URL built from the paired `ImageContent` block. Both go through the same checks: `new URL` succeeds, there is no username or password, a `data:` media type matches the item's `mime_type`, and an `https` URL's `origin` (scheme, host, and port) is exactly one entry in that server's `resourceDomains`. A prefix or substring check is not a match. Local declares none, so local HTML does not assign an `https` `src`.

Local, one inlined PNG from `fetch_outputs`. `preview_media` uses the same item for the contact sheet or waveform PNG it writes, and omits `prompt_id` when that file is not a job output:

```json
{
  "contract_version": 1,
  "items": [
    {
      "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC",
      "mime_type": "image/png",
      "kind": "image",
      "filename": "ComfyUI_00001_.png",
      "size_bytes": 69,
      "prompt_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    }
  ]
}
```

Cloud, one opted-in image. `url` is what the element plays. `download_url` is the shortlink. A batch entry may send `job_id` instead of `prompt_id`:

```json
{
  "contract_version": 1,
  "items": [
    {
      "url": "https://storage.googleapis.com/example-bucket/o/ComfyUI.png",
      "download_url": "https://cloud.comfy.org/api/s/example",
      "mime_type": "image/png",
      "kind": "image",
      "filename": "ComfyUI.png",
      "prompt_id": "7c1e0000-0000-4000-8000-000000000001"
    }
  ]
}
```

A real signed URL carries a query. That query stays in `url` for the element and is not copied into the text. `size_bytes` is omitted here because this entry has no measured length.

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

- Local text keeps today's metadata, including on-disk paths, and does not contain `data:` URLs. An image that already travels as `ImageContent` stays that block for hosts without Apps, and its item omits `url` and sets `image_block: true`. Paths stay, because the next tool call needs them. A `data:` URL for video or audio lives only on `items[].url`. The server does not also pretty-print `structuredContent` into a text block.
- Cloud text keeps today's link and download command, including an `/api/s/<id>` shortlink when one was minted.

`ImageContent` blocks pair with items in order. Walk `items` from the start. Each item with `image_block: true` takes the next image block. Any other item takes none, so a video, an item with a `url`, or an image past the inline cap (no `url`, no `image_block`) cannot shift the next image onto the wrong card. Nine image outputs with eight inlined therefore pair eight blocks with the eight marked items, and the ninth shows as not inlined. The block's `mimeType` and base64 `data` become `data:{mimeType};base64,{data}`, then the same checks as a `url`. If the number of `image_block` items and the number of image blocks differ, the viewer shows the text and does not pair a leftover block with a later item.

When the client advertised `io.modelcontextprotocol/ui` with MIME `text/html;profile=mcp-app`, cloud may append one line saying the viewer is showing the media and not to paste signed storage URLs. The line is extra. It does not remove the shortlink or the command. Other hosts get today's text unchanged.

A `contract_version` other than 1 still leaves this text in place. The iframe shows that text only, including when `results` or `outputs` is also present. It does not read version-1 names such as `filename` or `size_bytes`, and it does not guess a media element. When `contract_version` is absent or 1, an `items` value that is not an array does not opt in, and the viewer uses `results` or `outputs` when one of those arrays is present.

## Security

The document is one self-contained HTML string.

- DOM nodes come from `createElement`. Human-readable strings go through `textContent`.
- `img`, `video`, and `audio` `src` is the play URL after the checks above: `url`, the legacy `inline_url`, or the `data:` URL built from the paired `ImageContent` block. `download_url` is not a `src`.
- The bridge sends `ui/initialize` at protocol `2026-01-26`, then `ui/notifications/initialized`, then `ui/notifications/size-changed`. It handles `tool-result`, `host-context-changed` (`color-scheme`), and `ui/resource-teardown`.
- Messages go to `window.parent` and are accepted only when `event.source` is `window.parent`.
- Download may call `ui/open-link` only with `download_url`, or with `url` when the legacy rule above says `url` is the shortlink. Parse it with `new URL`. Reject a username or password. The `origin` must equal one of that server's own origins. For cloud, the path is exactly `/api/s/` plus an id of `[A-Za-z0-9_-]+`, and the query is empty or exactly `?raw=1`. A signed storage URL is never passed to `ui/open-link`. The save command stays in the text.

The document contains no `fetch`, `XMLHttpRequest`, `WebSocket`, `innerHTML`, `insertAdjacentHTML`, `document.write`, or `eval`. The local view makes no `tools/call`. The cloud exception is the read-only `get_output` call below. It requests no camera, microphone, geolocation, or clipboard permission, and it does not load `file://` or loopback.

CSP on the resource `_meta.ui` and on `resources/read`:

| Server | `connectDomains` | `resourceDomains` |
| --- | --- | --- |
| Local | omitted | omitted |
| Cloud | omitted | the session's public API origin, and the storage host its media URLs redirect to |

Local omits `ui.csp` entirely. The spec's restrictive-default block then allows `img-src` and `media-src` of `'self'` and `data:`, and sets `connect-src 'none'`. The same spec's CSP construction template still emits `connect-src 'self'` when `ui.csp` is absent, because `'self'` is in the template and the domain list is only appended. A present `csp` object with `connectDomains` omitted has the same split: the field text says no connections, and the template yields `connect-src 'self'`. Do not treat either omission as a guarantee of `connect-src 'none'`. The document itself opens no connection, whichever reading the host implements. The host template also allows `data:` when `resourceDomains` is set, so cloud can ignore a `data:` URL without a CSP change. No `frameDomains`, no `permissions`, no dedicated sandbox `domain`.

The host renders the HTML in a sandboxed iframe. Web hosts use the spec's outer sandbox proxy plus an inner frame. Inline script depends on the default `script-src 'self' 'unsafe-inline'`. The view assumes no other permission.

## Size limits

`_INLINE_IMAGE_MAX_BYTES` is 16 MiB of raw file bytes and `_INLINE_IMAGE_MAX_COUNT` is 8. Both already bound `fetch_outputs(inline_images=True)` and `preview_media`. Video and audio that join the inline result use this same budget. It is one budget for the whole tool result. `ImageContent` and a `data:` URL of the same file are not both sent, and that file counts once.

Inlining stays behind the flags that exist today. `fetch_outputs` puts bytes in `ImageContent` or a `data:` URL only when `inline_images=True` (the default is false). `preview_media` does so unless `inline_image=False`. An Apps-capable host does not turn inlining on, and a non-Apps host does not turn it off. `url_only=True` still downloads nothing, so it emits no inline bytes.

| Bound | Value | What counts |
| --- | --- | --- |
| One item | 16 MiB | that file's raw byte length |
| The result | 16 MiB | sum of raw bytes over `ImageContent` blocks and items that have a `data:` URL |
| The result | 8 items | those blocks and `data:` items, each file once |

`https` items do not count. A file over 16 MiB, one that would push the sum over 16 MiB, or a ninth inlined file is emitted without `url` and without an `ImageContent` block, in order, stopping at the first file that would break a bound. The on-disk file is unchanged. The text names the path and the size.

Base64 expands the wire size by about 4/3, so a full 16 MiB inline is on the order of 22 MiB of JSON. A host that truncates the tool result can drop the URL. The text fallback remains, and a missing or truncated `url` is shown as not inlined.

Cloud keeps its character ceiling for `https` results: `MAX_GET_OUTPUT_RESULT_CHARS` is 40,000 (text plus compact `structuredContent`), and one `get_output` page lists at most 25 files (`GET_OUTPUT_PAGE_MAX_FILES`). Files past that page are not entries in this result; the next call with `offset` returns them. The character ceiling is the only reason an entry stays and loses its URL: the server sets `url_omitted: true` and removes the URL strings, which is cloud's behavior today. The text says how to fetch the next page. Cloud does not embed `data:` URLs under this contract, so the 16 MiB cap applies there only if a later change inlines bytes.

## Expiry and refresh

Cloud mints two kinds of `https` media URL. A shortlink (`/api/s/<id>`) lasts about 5 minutes. A signed `storage.googleapis.com` URL lasts about 6 hours. Those are the lifetimes the cloud server mints today.

When a media element fails to load, the viewer replaces it with "Preview unavailable" through `textContent`. It does not `fetch` a replacement. On cloud, the iframe may call `get_output` with `{prompt_id}`, plus `offset` when the displayed result carries a non-zero `offset` (the read-only call below), which re-mints the links on the page it is showing. A legacy `results` or `outputs` result carries no `offset`, so the viewer cannot tell its page and offers no Refresh there; it shows "Preview unavailable" and the text, and the model refreshes with the `offset` the text names. The model does the same when the person or the viewer says the link expired: call `get_output` again, with `offset` when the text names one. Do not reuse a URL past the lifetime above, and do not paste the signed query into the reply.

## Cloud `wait_for_job` exception

Cloud may point `wait_for_job` at the same viewer. That result is status, not media. When the status says the job completed, the iframe may call `tools/call` for `get_output` with exactly `{prompt_id}`. Refresh after a media error may add the displayed page's `offset`, and nothing else. The call is read-only: no `save_to_disk`, no `confirm`, no other tool. Local does not do this.

Auto-load is not a hardcoded constant. The 2026-01-26 apps spec says a host may hold a view message for user approval, and `HostCapabilities` has no field that reports that policy. When the viewer cannot tell whether this host will forward `tools/call` without asking, it shows Load and sends `get_output` only after that control. When it knows the host forwards the call without approval, it may call immediately. Cloud's viewer currently auto-loads on every completion; that flag has to follow the host instead.

## Versioning

`contract_version` is an integer on the result. This document is version 1. The URI does not carry it. One `output-viewer` resource per server; the HTML reads the integer.

An additive change stays on version 1: a new optional field, or a new `kind` the current HTML shows as a filename row. Viewers ignore unknown fields. A breaking change (renaming `url`, changing what a `kind` means, adding a required field) bumps the integer. The server keeps emitting 1 until the HTML it serves understands the new number. A higher number the HTML does not understand shows only the text from `content`.

A new URI (`ui://comfy-mcp/output-viewer-v2` and the cloud twin) is for the case where the HTML and the payload must move together and a host could cache the old document against the new result. The HTML `appInfo.version` is not `contract_version`.

## Changes the cloud viewer needs to conform

Against PR 1894, PR 2005, and cloud `main` on 2026-11-14. PR 2005 only shares the bridge string. Keep `mcpAppBridgeJs` when the HTML changes.

### No change

- The URI stays `ui://comfy-cloud/output-viewer`.
- `results[]` and `outputs[]` stay the published agent API. The viewer keeps reading them until a response opts into `items`. Batch `summary`, `failed`, and `pending` stay extra fields.
- `url` plus `inline_url` stay the legacy play/download split. `inline_url` plays; a shortlink `url` downloads. `http:` stays rejected.
- `url_omitted: true` stays the character-ceiling signal. Do not delete the flag.
- A batch entry may keep `job_id`. The viewer accepts it as `prompt_id`.
- `size_bytes` stays omitted when the length is unknown, including when only `width` and `height` are known. Do not emit `0` for unknown.
- `wait_for_job` may keep `_meta.ui`. The read-only `get_output({prompt_id})` call from the iframe stays, including Refresh after a media error on an `items` result, which also sends the displayed page's `offset` when it is not 0.
- Filename extensions may keep classifying legacy entries. `kind` and `mime_type` are required only on an opted-in `items` entry.
- Security pins stay: no `fetch`, no `innerHTML`, text through `textContent`, no `connectDomains`, and the `chatgpt` profile stays resource-free. The upload widget, catalog picker, run-settings panel, and the iterate buttons stay outside this contract.
- `resourceDomains` stay this session's public API origin and the storage host those URLs redirect to (`https://storage.googleapis.com` today, or a narrower bucket origin when the server knows it). Do not add another environment's origin.

### Still to change

- Negotiate auto-load with the host. The viewer currently auto-loads on every completed `wait_for_job`. If the host requires approval for an app-initiated call, show Load and send `get_output({prompt_id})` only after that approval.
- Refresh on a legacy `results` or `outputs` result sends `{prompt_id}` today, which re-mints page one even when a later page is shown. Offer Refresh only on an `items` result, sending its `offset` when it is not 0.
- On a response that opts into `items`, set `url` to the play URL and `download_url` to the shortlink. Set `offset` on a page past the first, so Refresh re-mints that page. Do not also copy those URL strings into `results` or `outputs`. If the payload would pass 40,000 characters, omit `items` and stay on the legacy arrays.

## Non-goals

This file does not choose the local HTML, its tests, or the `mcp` package floor. It does not vendor the cloud viewer into this repo. A local implementation stays a thin MCP surface: no HTTP client, and no code copied from the cloud server.
