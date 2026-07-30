# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, **please report it responsibly**. Do **not** open a public GitHub issue.

### How to report

1. **GitHub Private Vulnerability Reporting (preferred)**
   Navigate to the [Security Advisories page](https://github.com/Comfy-Org/comfy-local-mcp/security/advisories/new) and submit a new advisory. This keeps the report confidential until a fix is available.

2. **Email**
   Send a detailed report to **support@comfy.org**. Include:
   - A description of the vulnerability and its potential impact
   - Steps to reproduce or a proof-of-concept
   - Affected versions (if known)
   - Any suggested fix or mitigation

### What to expect

| Step                                   | Timeline                                      |
| :------------------------------------- | :-------------------------------------------- |
| Acknowledgement of your report         | Within **3 business days**                    |
| Initial assessment and severity triage | Within **7 business days**                    |
| Patch or mitigation plan communicated  | Within **30 days** for critical/high severity |

We will keep you informed of our progress throughout the process. If the issue is accepted, we will coordinate disclosure with you and credit you in the advisory (unless you prefer to remain anonymous).

## Threat model

This section says what this server assumes, what it deliberately allows, and what it actually defends — so a reader can tell a real bug from working-as-intended. The short version: this is a thin wrapper that lets an agent drive a ComfyUI install running as you, on your machine. It is **not** a sandbox, and it does not try to be one. The interesting question is therefore not "can the agent touch my files" (it can) but "can a malicious *input* make the wrapper do something the agent was never given permission to do" — and that is what the guards below are for.

### Trust boundaries

**Trusted: the host environment and the MCP client registration.** `COMFY_BIN` (resolved on the `PATH` through `shutil.which`), `COMFYUI_URL` / `COMFYUI_HOST` / `COMFYUI_PORT` (see `_comfy_target`), and `COMFY_API_KEY` are read from the environment the user configured in their MCP client *before* the server starts. No tool can change any of them at runtime. An attacker who can set this process's environment — or edit the MCP client's config file — can already run arbitrary code as the user, so a malicious `COMFY_BIN` is **outside** this threat model: the server intentionally does not try to authenticate the binary it was told to run. It only checks that the binary exists, is readable, and is new enough (`_require_comfy_bin`, `_check_comfy_version`).

**Trusted: the local install.** comfy-cli, ComfyUI, and whatever custom nodes and models the user chose to install all run with the user's own privileges. This server is a wrapper over that install, not a sandbox around it.

**Semi-trusted: the agent / MCP client driving the tools.** Tool *inputs* are treated as adversarial — the working assumption is a prompt-injected agent. That assumption is what the argument guards (`_reject_nul`, `_reject_option_like`, `_guard_prompt_id`, `_guard_download_id`) and the consent gates exist for.

**Untrusted: content.** Workflow JSON files, registry and template metadata, workflow note text (`list_workflow_notes` — the README already says to treat it as data, not as instructions), and strings inside comfy-cli's result envelopes.

### What a compromised agent can do — by design

These are accepted capabilities, not defects. A prompt-injected agent holding this server's tools can do all of the following, mediated only by the agent host's own tool-permission UX:

- **Read and write files as the user.** `run_workflow(workflow_path)` and `validate_workflow(workflow_path)` read a path you name; `upload_file(paths)` copies any user-readable file into ComfyUI's `input` directory; `fetch_outputs(out_dir)`, `fetch_template(out_path)`, and `emit_partner_workflow(out_path)` write to a path you name. Paths are forwarded to comfy-cli verbatim once they pass the input-hygiene guards. The server does not sandbox the filesystem.
- **Execute arbitrary workflows.** A workflow is a program. With custom nodes installed, running one is arbitrary code execution inside the local ComfyUI's trust domain.
- **Control the ComfyUI lifecycle.** `launch_comfyui`, `stop_comfyui`, `restart_comfyui`, and `update_comfyui` start, stop, and mutate the install (`update_comfyui` runs a `git pull` plus a dependency reinstall). Note that `launch_comfyui(extra_args)` and `restart_comfyui(extra_args)` forward their arguments to ComfyUI **verbatim** after a `--` separator — including `--listen`, which exposes the unauthenticated ComfyUI API to the network.
- **Cause a local denial of service.** A workflow that passes every validation layer can still request an impossible total allocation and get the local ComfyUI process OOM-killed mid-run; this is documented in `run_workflow`'s `CAUTION` block.

### What this server defends against

- **Shell injection — structurally absent.** Every spawn is an argv list: `_run_comfy_raw` (`subprocess.Popen`), `_run_comfy_streaming` and `_start_login` (`asyncio.create_subprocess_exec`), and the `comfy --version` probe `_spawn_comfy_version`. `shell=True` appears nowhere in the repository. The three sites that run a real subcommand also pass `stdin=DEVNULL`, so a child can never read JSON-RPC protocol bytes out from under the client on this stdio transport; the `--version` probe reads no input and is bounded by its own timeout.
- **Argument and flag injection into comfy-cli.** Leading-dash positionals are rejected (`_reject_option_like` — e.g. `upload_file(paths=["--overwrite"])` is refused rather than silently becoming the overwrite flag), NUL bytes are rejected wherever they ride (`_reject_nul`), job and download handles are format-bounded (`_guard_prompt_id`, `_guard_download_id`), and `update_comfyui`'s `target` is validated against an allowlist so only the matched value — never the caller's raw string — reaches the command line.
- **Silent credit spend.** `partner_generate` requires comfy-cli's spend interlock and fails **closed** if it cannot prove the interlock is installed (`_require_spend_gate`). On a client that can be prompted, the human is elicited per call and the agent's own `confirm_spend=True` grants nothing; `confirm_spend` is only the documented fallback for a client that cannot show a prompt. `run_template` applies the same gate on the narrower terms the engine uses for it.
- **Silent destructive version switch.** `switch_comfyui_version` elicits the user on every call, and `confirm_switch=True` grants nothing on a client that can be prompted (`_resolve_switch_consent`). An unknown elicitation capability counts as *capable*, so a failed probe cannot demote a real client onto the caller's own say-so.
- **Model downloads escaping the models tree.** `download_model(relative_path)` is lexically traversal-checked and must name the `models` directory or a subfolder of it — specifically so a download cannot land in `custom_nodes/`, which ComfyUI would execute on its next start. **Stated limitation:** that check is lexical, so it cannot see a symlink or junction that *already* exists inside the models tree; if `models/link` already points at `custom_nodes`, then `models/link/pwn` passes. Planting such a link already requires write access to the models tree.
- **Resource exhaustion of the server itself.** Timeouts are bounded with hard ceilings (`_bounded_timeout`), output drains are capped to a trailing window (`_drain_capped_async`), a timeout kills the whole process tree rather than orphaning a `git pull` or a multi-GB transfer (`_kill_proc_tree`), and inline image previews are capped at 8 files / 16 MiB aggregate so a large batch cannot blow up a reply.

### What counts as a vulnerability

This server is a thin wrapper that shells out to the local [`comfy-cli`](https://github.com/Comfy-Org/comfy-cli) binary. The most relevant classes here are:

- Command injection or argument injection into the `comfy` invocation
- Path traversal or unsafe filesystem access via tool inputs (e.g. workflow paths, output directories, uploaded files)
- Exposure of secrets, credentials, or API keys
- Unsafe handling or parsing of comfy-cli's output that leads to code execution
- Denial-of-service against the MCP server

### Out of scope

- Defects within third-party software itself (comfy-cli, ComfyUI, custom nodes, MCP clients) — please report those to the respective maintainers. Vulnerabilities in **this repository's integration logic** — e.g. how tool inputs are turned into `comfy` arguments, or how comfy-cli's responses are validated and handled — are in scope and should be reported here.
- An attacker who already controls this server's environment, the MCP client's configuration, or the machine's user account. Each of those already implies arbitrary code execution as the user (see "Trust boundaries" above).
- Malicious custom nodes or models the user chose to install. Workflows execute in ComfyUI's trust domain, and this server does not sandbox them.
- The agent host's own permission and consent UX. This server raises elicitation wherever it can, but it cannot force a client to render a prompt; the code treats "cannot prompt" as a distinct case with documented `confirm_*` fallbacks whose defaults are the safe ones.
- Social engineering or phishing
- Reports from automated scanners without a demonstrated exploit

## Automated security tooling

GitHub's own safety nets are enabled on this repository, alongside the workflows in [`.github/workflows/`](.github/workflows/). What each one actually covers — so a reporter can tell what is already watched, and a contributor knows what will block a push:

- **Private vulnerability reporting** — the Security Advisories route in [How to report](#how-to-report) above. GitHub offers this on public repositories, so it is the preferred channel for a public release of this repository; the email route works either way.
- **Secret scanning, with push protection** — GitHub matches known provider credential patterns across this repository, and push protection **rejects a push that introduces one** rather than reporting it afterwards. This is a narrower net than the [TruffleHog workflow](.github/workflows/secret-scanning.yml), not a replacement for it: TruffleHog runs many more detectors (verified *and* unverified, including private keys and providers GitHub has no pattern for) but only after the push, on a diff gate plus a weekly full-history rescan. Push protection covers fewer patterns and stops the push itself. Neither subsumes the other, which is why both are on.
- **Dependabot alerts and security updates** — advisories against the `pip` and `github-actions` dependencies declared in [`.github/dependabot.yml`](.github/dependabot.yml), with fix PRs opened automatically. Alerts are independent of the version-update schedule in that file: a new advisory surfaces when it is published, not on the weekly/monthly cadence.

If a push of yours is rejected by push protection, treat the credential as burned — **rotate it**, then rewrite it out of your commits. Do not take the bypass option: bypassing on a public repository publishes the secret, and the rotation is needed regardless.

## Supported Versions

Security fixes are applied to the **latest release** on the `main` branch. We do not maintain patch branches for older versions.

## Disclosure Policy

We follow [coordinated disclosure](https://en.wikipedia.org/wiki/Coordinated_vulnerability_disclosure). We ask that you:

- Allow us reasonable time to investigate and address the issue before public disclosure.
- Act in good faith — do not access or modify other users' data.
- Do not exploit the vulnerability beyond what is necessary to demonstrate it.

We appreciate your help keeping this project and its users safe.
