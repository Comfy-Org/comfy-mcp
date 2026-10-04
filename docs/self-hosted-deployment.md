# Self-hosted remote MCP deployment

This is a simple home layout: one personal machine, ComfyUI and this fork side by side, reached by a remote agent through an HTTPS tunnel. It is a working homelab sketch, not a hardened production deployment. A public or multi-user service would need a tighter design than the one written down here.

Nginx on localhost is the piece this layout actually depends on. How that port is given an HTTPS name is interchangeable. Tailscale Funnel is one possible tunnel, described later as an example.

comfy-mcp does not open a public socket. `comfy-mcp` speaks MCP on stdio. `comfy-mcp-upload-server` accepts one streaming PUT on loopback. Every call that touches ComfyUI still goes through the `comfy` binary.

The templates under [`deploy/`](../deploy/) are the files to copy. They contain placeholders, not secrets.

## Architecture

```text
                        Internet
                           │
                           ▼
              HTTPS tunnel of your choice
           (Tailscale Funnel is one example)
                           │
                           ▼
                  Nginx 127.0.0.1:8191
                   │       │        │        │
                  /       /mcp    /upload/  /view?type=output
                   │       │        │        │
             Basic Auth   Bearer    one-time  read-only
                   │       │        upload    no Basic Auth
                   │       │        token     │
                   ▼       ▼        ▼        │
                ComfyUI  Supergateway  Upload server
                 :8189      :8190       :8192
                              │              │
                              ▼              │
                         comfy-mcp           │
                         (stdio)             │
                              │              │
                              ▼              ▼
                          comfy-cli ◄── temporary file
                              │
                              ▼
                         ComfyUI :8189
```

Why each piece exists:

| Piece | Role |
| --- | --- |
| ComfyUI | The engine. In this layout it stays on `127.0.0.1:8189`. It is not published directly. |
| comfy-cli | The only program that talks to ComfyUI. This server shells out to it. |
| comfy-mcp | MCP tools over stdio. It does not listen. |
| Supergateway | External process. Wraps that stdio server as Streamable HTTP on `127.0.0.1:8190/mcp`. Not part of this package. |
| comfy-mcp-upload-server | Receives a client's original bytes onto a temporary file, then stops. `complete_upload` runs `comfy upload`. It is not a ComfyUI client. |
| Nginx | The security and routing boundary on this machine. The only process an external tunnel should publish. |
| HTTPS tunnel | Optional, and outside this package. Gives Nginx a public name. Tailscale Funnel is one example among others. |

Remote download of generated files is a known gap. Today a homelab Nginx rule can leave `GET`/`HEAD /view?type=output` open so an agent can fetch a result. That shortcut is documented below because it is what this layout does now. A later version should replace it with a tighter download path. `type=input` and `type=temp` stay closed. There is no separate `/download` service and no base64 video path.

## Reference ports

These values are a known working set. Change them only together: the environment file, both units, the Nginx upstreams, and the tunnel target.

| Service | Bind |
| --- | --- |
| ComfyUI | `127.0.0.1:8189` |
| Supergateway (MCP) | `127.0.0.1:8190` |
| Nginx | `127.0.0.1:8191` |
| Upload server | `127.0.0.1:8192` |

Public name in the examples: `YOUR_HOSTNAME.example`. Replace it with the HTTPS name your tunnel actually serves. Do not publish `8189`, `8190`, or `8192`.

## Filesystem layout

```text
/opt/src/comfy-mcp/         source checkout
/opt/comfy-mcp/             Python virtualenv (not the source)
/var/lib/comfy-mcp/uploads  direct-upload spool
/var/log/comfy-mcp/         opt-in failure log directory
/etc/comfy-mcp.env          runtime environment (not in git)
/opt/ComfyUI                 example ComfyUI checkout, separate from this package
```

`/opt/comfy-mcp` is the virtualenv created by `python3 -m venv`. The source stays under `/opt/src/comfy-mcp`. Mixing the two makes upgrades and `pip install -e` point at the wrong tree.

`/opt/ComfyUI` is the recommended place for the ComfyUI checkout itself: a system path, separate from a home directory and from this package's virtualenv. This package does not create it. Point ComfyUI's own launch at `127.0.0.1` and port `8189` (its `--listen` / `--port`). This server reaches it through `COMFY_LOCAL_URL`, not by importing that tree.

### Service user

The example units run as an unprivileged user named `comfy`, not as root.

```bash
sudo useradd --system --user-group \
  --home-dir /var/lib/comfy-mcp \
  --shell /usr/sbin/nologin \
  comfy
```

That user must be able to execute `/opt/comfy-mcp/bin` and read `/opt/src/comfy-mcp`. It owns the spool and the log directory. If you use another account, change `User=` and `Group=` in both units and the `chown` commands below.

## Python installation

Debian-style packages: `python3` and `python3-venv`. Node.js and npm are required only because Supergateway is a Node program.

```bash
sudo mkdir -p /opt/src
sudo git clone https://github.com/Comfy-Org/comfy-mcp.git /opt/src/comfy-mcp
# A fork: clone your fork, or rsync a checkout into that path. See "Updating".

sudo python3 -m venv /opt/comfy-mcp
sudo /opt/comfy-mcp/bin/pip install -U pip
sudo /opt/comfy-mcp/bin/pip install -e /opt/src/comfy-mcp
sudo /opt/comfy-mcp/bin/pip install "comfy-cli>=1.14.0"
```

Editable install (`-e`) is the convenient development deploy: source edits are picked up on process restart, without reinstalling, until `pyproject.toml`, dependencies, or entry points change.

A non-editable install from the same checkout:

```bash
sudo /opt/comfy-mcp/bin/pip install /opt/src/comfy-mcp
sudo /opt/comfy-mcp/bin/pip install "comfy-cli>=1.14.0"
```

A non-editable install copies the package into the virtualenv. Source edits do nothing until you run `pip install` again and restart.

This package's console scripts are only:

```text
/opt/comfy-mcp/bin/comfy-mcp
/opt/comfy-mcp/bin/comfy-mcp-upload-server
```

`/opt/comfy-mcp/bin/comfy` appears after the separate `comfy-cli` install. It is not an entry point of `comfy-mcp`. `COMFY_BIN` must name that binary. comfy-cli is deliberately not a dependency of this package; installing `comfy-mcp` alone does not provide `comfy`.

Check the two programs that implement `--version`. Do not start `comfy-mcp-upload-server` by hand here; with no unit it listens on `127.0.0.1:8192` until stopped. The systemd section starts it.

```bash
/opt/comfy-mcp/bin/comfy-mcp --version
/opt/comfy-mcp/bin/comfy --version
test -x /opt/comfy-mcp/bin/comfy-mcp-upload-server
```

Give the service user read and execute rights on the virtualenv and the source. A typical root-owned `755`/`644` tree is enough. Do not mode the virtualenv `0700` unless `comfy` owns it.

## Environment

Copy [`deploy/comfy-mcp.env.example`](../deploy/comfy-mcp.env.example) to `/etc/comfy-mcp.env` and edit it there.

```bash
sudo install -m 0640 -o root -g comfy \
  deploy/comfy-mcp.env.example /etc/comfy-mcp.env
sudoedit /etc/comfy-mcp.env
```

| Variable | Who reads it | What it does |
| --- | --- | --- |
| `COMFY_BIN` | comfy-mcp | Absolute `comfy` binary. Default is the bare name `comfy` on `PATH`. |
| `COMFY_LOCAL_URL` | comfy-cli, via the inherited environment | Local ComfyUI when it is not `127.0.0.1:8188`. Example: `http://127.0.0.1:8189`. This server does not parse the variable. |
| `COMFY_MCP_UPLOAD_DIR` | upload session | Spool directory. Default `/var/lib/comfy-mcp/uploads`. |
| `COMFY_MCP_UPLOAD_PUBLIC_BASE_URL` | `init_upload` | HTTPS origin only: no userinfo, path, query, or fragment. Required for a manual PUT session. |
| `COMFY_MCP_UPLOAD_TTL_SECONDS` | upload session | Lifetime. Default `600`. Clamped to 30..86400. |
| `COMFY_MCP_MAX_UPLOAD_MB` | upload session | Max original size in megabytes. Default `1024`. Values above `8192` are clamped. |
| `COMFY_MCP_UPLOAD_HOST` | upload server | Bind address. Only `127.0.0.1` or `::1`. Anything else refuses to start. Default `127.0.0.1`. |
| `COMFY_MCP_UPLOAD_PORT` | upload server | Bind port. Default `8192`. |
| `COMFY_MCP_DEBUG_LOG` | comfy-mcp | Opt-in JSONL failure log. Unset, empty, or `0` disables it. Any other value is the file path. |
| `COMFY_MCP_GIT_SHA` | both processes | Optional 7..40 hex digits logged once at startup when the tree has no `.git`. |

Do not set `COMFYUI_URL` to the public hostname. That variable selects a *remote* ComfyUI and forwards `--host`/`--port`. On this box the ComfyUI is local; `COMFY_LOCAL_URL` is the knob.

Leave `COMFY_API_KEY`, Basic Auth passwords, and the MCP Bearer out of this file. The MCP Bearer lives only in the Nginx config on the server. The one-time upload token is minted per session and is not a configuration value.

## Spool

```bash
sudo mkdir -p /var/lib/comfy-mcp/uploads
sudo chown comfy:comfy /var/lib/comfy-mcp /var/lib/comfy-mcp/uploads
sudo chmod 700 /var/lib/comfy-mcp /var/lib/comfy-mcp/uploads
```

The spool holds an in-progress upload: a manifest and, after a successful PUT, the sanitized filename. `complete_upload` deletes the session directory after `comfy upload` succeeds. A failed PUT returns the session to `initialized` so the same token can be retried until expiry. Abandoned sessions are removed on the next upload call once `COMFY_MCP_UPLOAD_TTL_SECONDS` has elapsed. A session left in `completing` is kept until that extended expiry so an in-flight `comfy upload` is not deleted under itself.

This directory is not ComfyUI's `input` folder. ComfyUI learns the file only after `comfy upload`, under the name `comfy_filename`.

## MCP process and Supergateway

Transport:

```text
comfy-mcp stdio
    ↓
Supergateway
    ↓
Streamable HTTP
127.0.0.1:8190/mcp
```

Supergateway is the npm package [`supergateway`](https://www.npmjs.com/package/supergateway). This repository does not pin a version. Install a current one and read `supergateway --help` before relying on a flag.

```bash
sudo npm install -g supergateway
command -v supergateway
supergateway --help
```

The default listen address is every interface. The unit sets `--host 127.0.0.1`. If `command -v` is not `/usr/local/bin/supergateway`, edit `ExecStart` in the unit.

The public MCP URL is:

```text
https://YOUR_HOSTNAME.example/mcp
```

Nginx checks the permanent Bearer, then proxies to `127.0.0.1:8190`. Clients configure that HTTPS URL. They do not dial port 8190.

`--stateful` is optional. Add it only if a client needs a persistent Streamable HTTP session; the reference unit is stateless. Do not put the MCP Bearer in `--apiKey`: that would duplicate the Nginx gate and place the secret on the process command line.

## systemd

Copy the examples:

```bash
sudo cp deploy/systemd/comfy-mcp.service.example \
  /etc/systemd/system/comfy-mcp.service
sudo cp deploy/systemd/comfy-mcp-upload.service.example \
  /etc/systemd/system/comfy-mcp-upload.service
sudo systemctl daemon-reload
sudo systemctl enable --now comfy-mcp
sudo systemctl enable --now comfy-mcp-upload
```

Both units set `EnvironmentFile=/etc/comfy-mcp.env`, `WorkingDirectory=/opt/src/comfy-mcp`, `User=comfy`, `UMask=0077`, and `Restart=on-failure`.

`comfy-mcp.service` runs Supergateway. `comfy-mcp-upload.service` runs `/opt/comfy-mcp/bin/comfy-mcp-upload-server`, which binds `127.0.0.1:8192` from the environment file.

Check:

```bash
systemctl status comfy-mcp --no-pager
systemctl status comfy-mcp-upload --no-pager
```

Each process logs one line at startup:

```text
startup process=comfy-mcp version=... git=...
startup process=comfy-mcp-upload-server version=... git=...
```

Different `version` or `git` values mean the two processes are not the same build.

Both processes are long-running interpreters that imported this package at start. After a deploy of shared Python code, restart both:

```bash
sudo systemctl restart comfy-mcp-upload comfy-mcp
```

A parser-only change is loaded by the MCP process, but restarting both is the safe ordinary deploy. Restarting one leaves the other on the previous import.

## Nginx

Install the example outside the git checkout and replace the Bearer placeholder there:

```bash
sudo cp deploy/nginx/comfy-gateway.conf.example \
  /etc/nginx/sites-available/comfy-gateway.conf
sudoedit /etc/nginx/sites-available/comfy-gateway.conf
sudo ln -s /etc/nginx/sites-available/comfy-gateway.conf \
  /etc/nginx/sites-enabled/comfy-gateway.conf
```

Create the UI password file. It is not in this repository.

```bash
sudo htpasswd -c /etc/nginx/.htpasswd YOUR_UI_USER
sudo chmod 640 /etc/nginx/.htpasswd
```

The file defines four routes. `location = /view` is an exact match, so it wins over `location /`. The `^~` prefixes win over the UI prefix as well.

### MCP

`location ^~ /mcp` returns 401 unless `Authorization` is exactly `Bearer REPLACE_WITH_MCP_BEARER`, then proxies to `127.0.0.1:8190` with buffering off. Replace the placeholder in `/etc/nginx`, not in `deploy/`. The comparison is a single `return` inside `if`, which is the reference gate. The secret must not be committed.

### Direct binary upload

```nginx
location ^~ /upload/ {
    proxy_pass http://127.0.0.1:8192;
    proxy_http_version 1.1;
    proxy_request_buffering off;
    proxy_buffering off;
    client_max_body_size 0;
    proxy_connect_timeout 60s;
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
}
```

This prefix has no Basic Auth and no permanent MCP Bearer. `comfy-mcp-upload-server` checks the one-time Bearer from `init_upload`. The client sends it in `Authorization`, never in the URL. Buffering is off so a large body is streamed to disk instead of being collected first. `curl --upload-file` sends `Content-Length`, which the server requires and must match the size declared to `init_upload`.

`GET /healthz` exists on the upload server for local checks (`curl -fsS http://127.0.0.1:8192/healthz`). It is not under `/upload/`, so the public proxy does not expose it.

### Generated outputs

This route is a known shortcut. A later version should replace public output fetches with a tighter download path. Until then, a home Nginx can allow the request only when the query parameter `type` is `output`. Any other value, including `input`, `temp`, or a missing type, is 403. Basic Auth is off for that one route. `limit_except GET HEAD` denies every other method, and HEAD is included so `curl -I` behaves like GET.

Anyone who obtains the exact public output URL can fetch that file while ComfyUI still has it. The route does not list directories. Treat the URL as an unauthenticated link. It is not signed.

### ComfyUI UI

`location /` requires Basic Auth (`auth_basic_user_file /etc/nginx/.htpasswd`) and proxies to `127.0.0.1:8189`, including the WebSocket upgrade headers ComfyUI's UI uses (`Upgrade` and `Connection`). That password is independent of the MCP Bearer and of the one-time upload token.

## Nginx validation

```bash
sudo nginx -t
sudo systemctl reload nginx
```

UI, still protected:

```bash
curl -I https://YOUR_HOSTNAME.example/
```

Expected: `401` without credentials.

Input, still private:

```bash
curl -I \
  'https://YOUR_HOSTNAME.example/view?filename=test.png&type=input'
```

Expected: `403`.

Temp, still private:

```bash
curl -I \
  'https://YOUR_HOSTNAME.example/view?filename=test.png&type=temp'
```

Expected: `403`.

Output, using a file that already exists under ComfyUI's output folder:

```bash
curl -I \
  'https://YOUR_HOSTNAME.example/view?filename=FILE.mp4&subfolder=video&type=output'
```

Expected: `200`.

Invalid upload:

```bash
curl -i \
  -X PUT \
  --data-binary test \
  https://YOUR_HOSTNAME.example/upload/not-a-real-upload
```

Expected: an HTTP error, never a successful store.

MCP without the permanent Bearer:

```bash
curl -i https://YOUR_HOSTNAME.example/mcp
```

Expected: `401`.

## Publishing Nginx with HTTPS

The layout stops at Nginx on `127.0.0.1:8191`. Putting that port on the Internet is a separate choice. A reverse proxy, a VPN, or an HTTPS tunnel all fit. Tailscale Funnel is one simple option for a home machine. It is not part of comfy-mcp, and it is not a production requirement.

Example, if you use Tailscale Funnel:

```bash
sudo tailscale funnel --https=443 --bg http://127.0.0.1:8191
sudo tailscale funnel status
```

Expected shape:

```text
https://YOUR_HOSTNAME.example
|-- / proxy http://127.0.0.1:8191
```

With Funnel, the public name looks like `name.tailXXXX.ts.net`. The tunnel should publish Nginx only. ComfyUI, Supergateway, and the upload server stay on localhost. Routing and authentication stay in Nginx. Another tunnel that targets `http://127.0.0.1:8191` replaces this example entirely.

`COMFY_MCP_UPLOAD_PUBLIC_BASE_URL` must be the HTTPS origin clients use (`https://YOUR_HOSTNAME.example`), because `init_upload` prints that origin into the curl command.

## Direct upload (Claude, Cursor, other local agents)

Bytes do not travel through MCP JSON.

```text
local attachment
    ↓
init_upload(filename, file_size, mime_type, overwrite)
    ↓
upload_url + one-time Authorization header + curl_command
    ↓
client HTTP PUT of the original bytes
    ↓
server checks size and SHA-256, renames to the sanitized filename, state=ready
    ↓
complete_upload(upload_id)
    ↓
comfy upload
    ↓
ComfyUI input
    ↓
comfy_filename
```

On the machine that has the file:

```bash
wc -c < "$FILE" | tr -d ' '
file --mime-type -b "$FILE"
```

Call `init_upload` with that basename, byte count, and MIME type. Then export `FILE` to the same path and run the returned `curl_command`. The command is a PUT with `--upload-file "$FILE"` and `Authorization: Bearer` set to the one-time token. Then call `complete_upload` with the returned `upload_id`.

The next workflow must load `comfy_filename` from the `upload_complete` result. That value is `uploads[].cloud_name` from comfy-cli. ComfyUI may rename on collision; do not substitute the client path, `/mnt/user-data/uploads/...`, the spool path, or the source basename when it differs from `comfy_filename`.

An expected failure returns `kind: "upload_error"` with `stage` and `message`. The token is stored only as a SHA-256 hash and is cleared after a successful PUT.

## ChatGPT native file

When the host honors tool metadata, `init_upload` declares `_meta["openai/fileParams"] = ["file"]`. ChatGPT can pass its native file object as `init_upload(file=...)`.

```text
host file (download_url, file_id, optional file_name)
    ↓
HTTPS stream onto a local file whose basename is the sanitized name
    ↓
comfy upload
    ↓
kind=upload_complete and comfy_filename
```

No curl step in that mode. The download is HTTPS-only, with checks against private and loopback targets, and the signed URL is not written to the client error or the upload logs. This is the ChatGPT path. Other MCP hosts do not automatically gain it by calling the same tool without a host file; they use the manual PUT flow above.

`upload_file(paths=[...])` remains the tool for a file that is already on the server filesystem.

## Output retrieval

Public download of generated files is a known limitation of this home layout. ComfyUI already serves the file, and the Nginx exception above leaves `type=output` readable without Basic Auth so a remote agent can open the URL. That is enough for a personal server. A later version should add a proper download path, with access control tighter than "anyone who has the URL". There is still no custom `/download` service and no base64 transfer in this package.

ComfyUI's route:

```text
/view?filename=...&subfolder=...&type=output
```

Two different clients must not be confused:

| Caller | Path | Auth |
| --- | --- | --- |
| `fetch_outputs` / `comfy download` on the server | comfy-cli to local ComfyUI | Not affected by Nginx Basic Auth. It never goes through the public proxy. |
| A remote agent or browser opening the public URL | Nginx, then ComfyUI | Needs the `/view` `type=output` exception. Without it, Basic Auth on `location /` returns 401. |

`run_workflow` / `job` results may already contain `/view` URLs. A remote client can open those only when Nginx allows `type=output`. Rewrite a loopback host in that URL to the public HTTPS origin if the client is not on the server.

## Developer note: stdout parser

Not a networking failure. comfy-cli may print one JSON document that is valid for `json.loads` and that Python `str.splitlines()` still splits, because JSON whitespace or a character inside a string is a line break for Python. `workflow slots` has produced that shape: exit code 0, a parseable envelope, and a wrapper that used to report no JSON.

`_last_json_object` parses the stripped stdout as one document first and falls back to NDJSON only when that is not a single object. In the failure log, `exit_code` 0 together with `kind` `no_json` means the wrapper still did not accept the document. Inspect the log. Do not start by changing ports or the HTTPS tunnel.

## Debug logging

In `/etc/comfy-mcp.env`:

```dotenv
COMFY_MCP_DEBUG_LOG=/var/log/comfy-mcp/failures.jsonl
```

```bash
sudo mkdir -p /var/log/comfy-mcp
sudo chown comfy:comfy /var/log/comfy-mcp
sudo chmod 700 /var/log/comfy-mcp
sudo systemctl restart comfy-mcp
```

The log records comfy-cli failures only. Successful calls write nothing. Unset, empty, or `0` turns it off and creates no directory.

```bash
sudo tail -n 20 /var/log/comfy-mcp/failures.jsonl | jq .
sudo journalctl -u comfy-mcp -n 100 --no-pager
sudo journalctl -u comfy-mcp-upload -n 100 --no-pager
```

These logs must not contain the permanent MCP Bearer, a one-time upload Bearer, an `Authorization` header, a signed host-file URL, or file bytes. The failure log masks credentials inside URLs. Journal lines for upload carry `upload_id`, stage, and state, not the token.

## Deployment checklist

1. Clone or copy the source to `/opt/src/comfy-mcp`.
2. Create the virtualenv at `/opt/comfy-mcp`.
3. `pip install -e /opt/src/comfy-mcp` and `pip install "comfy-cli>=1.14.0"` with that virtualenv's pip.
4. Install `/etc/comfy-mcp.env` from the example and set `COMFY_BIN`, `COMFY_LOCAL_URL`, and `COMFY_MCP_UPLOAD_PUBLIC_BASE_URL`.
5. Create `/var/lib/comfy-mcp/uploads` mode `0700`, owned by `comfy`.
6. Copy both systemd units into `/etc/systemd/system/` and `systemctl daemon-reload`.
7. `systemctl enable --now comfy-mcp` and `systemctl enable --now comfy-mcp-upload`.
8. Copy the Nginx example to `/etc/nginx`, replace the Bearer placeholder, and enable the site.
9. Create `/etc/nginx/.htpasswd` for the UI.
10. Confirm the MCP Bearer in the Nginx file matches what remote clients will send, and that the file is not in git.
11. Point the HTTPS tunnel at `http://127.0.0.1:8191`. Tailscale Funnel is one example, above.
12. Check the four loopback listeners (`ss` below).
13. Run the public curl checks (401, 403, output 200, rejected PUT).
14. Connect the MCP client to `https://YOUR_HOSTNAME.example/mcp` with the permanent Bearer.
15. Run `init_upload`, the returned curl, and `complete_upload`. Confirm `comfy_filename`.
16. Run one small workflow that loads that `comfy_filename`.
17. Fetch the resulting `/view?...&type=output` URL from outside the machine and confirm `type=input` still returns 403.

## Verification matrix

```bash
sudo ss -lntp | grep -E '8189|8190|8191|8192'
```

Each of `8189`, `8190`, `8191`, and `8192` should show `127.0.0.1` (or `::1` if you chose that for the upload server). `0.0.0.0` or `*` on 8190 means Supergateway was started without `--host 127.0.0.1`.

| Check | Expected |
| --- | --- |
| `127.0.0.1:8189` | ComfyUI |
| `127.0.0.1:8190` | Supergateway / MCP |
| `127.0.0.1:8191` | Nginx |
| `127.0.0.1:8192` | `comfy-mcp-upload-server` (`GET /healthz` returns `ok`) |
| `/` without Basic Auth | 401 |
| `/mcp` without the permanent Bearer | 401 |
| `PUT /upload/<invalid>` | rejected |
| `/view?...&type=input` | 403 |
| `/view?...&type=temp` | 403 |
| `/view?...&type=output` for a file that exists | 200 |
| `init_upload` → PUT → `complete_upload` | `kind=upload_complete` and `comfy_filename` |

## Updating from a development machine

From the checkout you edit:

```bash
rsync -av --delete \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='venv/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='.DS_Store' \
  ./ \
  USER@SERVER:/opt/src/comfy-mcp/
```

On the server, as a user who may write the virtualenv:

```bash
/opt/comfy-mcp/bin/pip install -e /opt/src/comfy-mcp
sudo systemctl restart comfy-mcp-upload comfy-mcp
systemctl status comfy-mcp --no-pager
systemctl status comfy-mcp-upload --no-pager
```

An editable install already sees ordinary `.py` edits after a restart. Run `pip install -e` again when dependencies, package metadata, entry points, or `pyproject.toml` change. During active development, running the editable install before every restart is the simple safe sequence.

Confirm the two startup lines share `version` and `git`. Set `COMFY_MCP_GIT_SHA` in `/etc/comfy-mcp.env` when the server tree has no `.git` directory (the rsync above excludes it).

## Security

- Bind ComfyUI, Supergateway, and the upload server to localhost. The upload server refuses any other bind.
- Expose only Nginx (or the equivalent proxy), and only over HTTPS.
- Do not commit the MCP Bearer, the htpasswd file, `/etc/comfy-mcp.env`, or one-time upload tokens.
- MCP uses a permanent Bearer at Nginx. Upload uses a different one-time Bearer checked by the upload server. The UI uses Basic Auth. The three are not interchangeable.
- Public `GET`/`HEAD` of `/view?type=output` is a known homelab shortcut. An exact URL is readable by anyone who has it. A later version should replace that with a tighter download path.
- `/view?type=input` and `/view?type=temp` stay blocked.
- The spool is mode `0700` and owned by the service user. `UMask=0077` applies to both units.
- Do not publish ports 8189, 8190, or 8192 on the Internet or on the LAN.
- Local ComfyUI has no authentication of its own. The bind address plus Nginx are the boundary.
- This is a simple home layout, not a production deployment. A single-user stdio `comfy-mcp` in an MCP client's config remains the local setup described in the README.

## Troubleshooting

### comfy-cli tries port 8188

Symptom:

```text
cannot reach http://127.0.0.1:8188/object_info
```

A manual shell does not load `/etc/comfy-mcp.env`. `COMFY_LOCAL_URL` is unset, so comfy-cli uses its default port.

```bash
COMFY_LOCAL_URL=http://127.0.0.1:8189 \
  /opt/comfy-mcp/bin/comfy --json --where local env
```

The systemd units load the file via `EnvironmentFile`. If the service still uses 8188, the file is missing the variable or the unit was not restarted after the edit.

### PUT succeeds and `complete_upload` fails

Restart both services after a shared-code deploy. Compare the two `startup process=...` lines. A mismatch means one process is still on the previous import.

```bash
sudo systemctl restart comfy-mcp-upload comfy-mcp
```

Then start a new `init_upload`. Do not reuse an expired or half-written session.

### The file lands in ComfyUI as `payload.ready`

That name is an obsolete spool filename. The current server renames the body to the sanitized basename before `comfy upload`, and `complete_upload` passes that path. An upload process that still writes `payload.ready` is running old code.

Restart both units and create a new session. Old sessions are not migrated; they expire.

### `no_json` with exit code 0

Open `COMFY_MCP_DEBUG_LOG` and look at `kind`, `exit_code`, and the shape line in the journal (`whole_json_valid`, `top_level_type`, `json_type`, `schema`). Exit code 0 with `kind` `no_json` is the wrapper failing to accept stdout, not a dead ComfyUI port. The current parser tries `json.loads` on the whole document before the NDJSON fallback. See the developer note above.

### Output URL returns 401

`location = /view` is missing, or it is not the exact match shown in the example, so the request falls through to `location /` and Basic Auth. Reload Nginx after fixing the site file. `type=output` must be the query parameter ComfyUI uses, not a path segment.

### Confirm input and temp stay closed

After any Nginx edit, repeat the `type=input` and `type=temp` checks. Both must stay 403. A broad `auth_basic off` on `location /` would expose the UI and those file types; do not do that.
