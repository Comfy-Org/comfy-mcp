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

### What counts as a vulnerability

This server is a thin wrapper that shells out to the local [`comfy-cli`](https://github.com/Comfy-Org/comfy-cli) binary. The most relevant classes here are:

- Command injection or argument injection into the `comfy` invocation
- Path traversal or unsafe filesystem access via tool inputs (e.g. workflow paths, output directories, uploaded files)
- Exposure of secrets, credentials, or API keys
- Unsafe handling or parsing of comfy-cli's output that leads to code execution
- Denial-of-service against the MCP server

### Out of scope

- Defects within third-party software itself (comfy-cli, ComfyUI, MCP clients) — please report those to the respective maintainers. Vulnerabilities in **this repository's integration logic** — e.g. how tool inputs are turned into `comfy` arguments, or how comfy-cli's responses are validated and handled — are in scope and should be reported here.
- Social engineering or phishing
- Reports from automated scanners without a demonstrated exploit

## Supported Versions

Security fixes are applied to the **latest release** on the `main` branch. We do not maintain patch branches for older versions.

## Disclosure Policy

We follow [coordinated disclosure](https://en.wikipedia.org/wiki/Coordinated_vulnerability_disclosure). We ask that you:

- Allow us reasonable time to investigate and address the issue before public disclosure.
- Act in good faith — do not access or modify other users' data.
- Do not exploit the vulnerability beyond what is necessary to demonstrate it.

We appreciate your help keeping this project and its users safe.
